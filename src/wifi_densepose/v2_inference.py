"""CPU inference for the three InSense V2 models on one 64-frame CSI window."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .models.cnn import SmallCSIPoseCNN
from .models.insense_heads import MotionOrPostureTCN

WINDOW = 64
K4 = ('B0_UPRIGHT_ACTIVITY', 'B1_LOWER_BODY_ACTIVITY', 'B2_BENT_OR_LOW', 'B3_JUMP')
M3 = ('S0_STATIC', 'S1_IN_PLACE_ACTIVITY', 'S3_POSTURE_TRANSITION')
STATE_NAMES = {
    'S00': 'Upright Static', 'S01': 'Upright Activity', 'S02': 'Upright Transition',
    'S10': 'Lower-body Static', 'S11': 'Lower-body Activity', 'S12': 'Lower-body Transition',
    'S20': 'Bent or Low Static', 'S21': 'Bent or Low Activity', 'S22': 'Bent or Low Transition',
    'S30': 'Jump',
}
CHECKPOINTS = {
    'root': ('267314703a410c8d6d6c7dd95911fb30fad39c0b8c83fe539538d494c5d8748b', 95827, 1),
    'posture': ('14d1a1f7681524defab98786031af866c473ea3487a765b7e3d199bd25b9e2da', 111492, 15),
    'motion': ('4d7b078bb690b406b695db88bc8153521d5fe3c92c6031488e2a343ea629007e', 115523, 4),
}
K4_SHA = '28d8f653c4cbba935e9ed2a2e5888d98f2d4e85a96365f65430d4118850559b9'
M3_SHA = '74f6a4414761350d9b82f1bf624b5ae6885c96210526ff9b7e965c48af47e3f6'


@dataclass(frozen=True)
class WindowID:
    subject: str
    sequence: str
    window: str
    frames: tuple[int, ...]

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.subject, self.sequence, self.window)):
            raise ValueError('Missing window identity')
        frames = tuple(self.frames)
        if (len(frames) != WINDOW or any(not isinstance(f, Integral) or isinstance(f, bool) or f < 0 for f in frames)
                or frames != tuple(range(int(frames[0]), int(frames[0])+WINDOW))):
            raise ValueError('Expected 64 contiguous ordered frame IDs')
        object.__setattr__(self, 'frames', tuple(map(int,frames)))


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim == 0 or not x.size or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid logits')
    x = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(x)
    return weights / weights.sum(axis=-1, keepdims=True)


def head_output(logits, classes):
    p = probabilities(logits)
    if p.shape != (len(classes),):
        raise ValueError('Unexpected class dimension')
    index = int(np.argmax(logits))
    return index, {
        'predicted_class': classes[index],
        'class_probabilities_pct': {
            name: 100 * float(value) for name, value in zip(classes, p)
        },
    }


def state_code(posture, motion=None):
    if type(posture) is not int or posture not in range(4):
        raise ValueError('Invalid posture class')
    if posture == 3:
        return 'S30'
    if type(motion) is not int or motion not in range(3):
        raise ValueError('Invalid motion class')
    return f'S{posture}{motion}'


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg.get('schema') != 'insense-v2/1':
        raise ValueError('Unsupported V2 inference configuration')
    if cfg.get('checkpoints') != {name: terms[0] for name,terms in CHECKPOINTS.items()}:
        raise ValueError('Checkpoint allowlist mismatch')
    for name, classes, digest in [('posture', K4, K4_SHA), ('motion', M3, M3_SHA)]:
        h = cfg['heads'][name]
        if tuple(h['classes']) != classes or h['taxonomy_sha256'] != digest:
            raise ValueError('Wrong class order or taxonomy digest')
        for key in ('mean','std'):
            v = np.asarray(h[key], dtype=np.float64)
            if v.shape != (48,) or not np.isfinite(v).all() or (key=='std' and np.any(v <= 0)):
                raise ValueError('Invalid pose normalization')
    return cfg


def preprocess_csi(values, *, stage7=False):
    if np.iscomplexobj(values):
        raise ValueError('Expected real CSI amplitudes')
    x = np.array(values, dtype=np.float64, copy=True)
    if x.shape != (WINDOW,3,114,10):
        raise ValueError('Expected [64,3,114,10] CSI tensor')
    if stage7:
        if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError('Invalid Stage7 amplitudes')
        return x.astype(np.float32)
    for frame in x:
        for k in range(10):
            plane = frame[:,:,k]
            finite = np.isfinite(plane)
            if not finite.any():
                raise ValueError('All missing CSI slice')
            plane[~finite] = plane[finite].mean()
        lo, hi = frame.min(), frame.max()
        if hi <= lo or not np.isfinite(hi-lo):
            raise ValueError('Degenerate raw CSI frame')
        frame[:] = (frame-lo)/(hi-lo)
    return x.astype(np.float32)


def checkpoint_state(path, name):
    payload = Path(path).read_bytes()
    digest, _, epoch = CHECKPOINTS[name]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'{name} checkpoint SHA256 mismatch')
    # The exact trusted project artifact can include optimizer and NumPy RNG metadata.
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    if state['epoch'] != epoch:
        raise ValueError(f'{name} checkpoint epoch mismatch')
    if name == 'root':
        if state['config']['arm'] != 'D0': raise ValueError('Expected the D0 arm')
    else:
        expected_taxonomy = K4_SHA if name=='posture' else M3_SHA
        if (state['config']['root_sha256'] != CHECKPOINTS['root'][0]
                or state['config']['taxonomy_sha256'] != expected_taxonomy):
            raise ValueError('Root or taxonomy binding mismatch')
        order = state['taxonomy']['class_order'] if name=='posture' else state['config']['classes']
        if tuple(order) != (K4 if name=='posture' else M3):
            raise ValueError('Checkpoint class order mismatch')
    return state


class InSenseV2:
    def __init__(self, config_path, *, root, posture, motion):
        self.config=load_config(config_path)
        self.root=SmallCSIPoseCNN()
        self.posture=MotionOrPostureTCN(False)
        self.motion=MotionOrPostureTCN(True)
        for name,path in [('root',root),('posture',posture),('motion',motion)]:
            state=checkpoint_state(path,name)
            model=getattr(self,name)
            model.load_state_dict(state['model_state_dict' if name=='posture' else 'model'],strict=True)
            if sum(p.numel() for p in model.parameters()) != CHECKPOINTS[name][1]:
                raise ValueError('Model parameter count mismatch')
            model.eval().requires_grad_(False)

    @torch.inference_mode()
    def logits_from_pose(self, poses):
        """Compute K4 and M3 logits for the same batch of D0 pose windows."""
        x = np.asarray(poses)
        if x.ndim != 4 or x.shape[1:] != (WINDOW,17,3) or not len(x) or not np.isfinite(x).all():
            raise ValueError('Expected finite [N,64,17,3] D0 poses')
        # The heads were trained on joints 1–16 in the original D0 coordinates.
        flat=x[:,:,1:,:].reshape(len(x),WINDOW,48).transpose(0,2,1)
        answer={}
        for name in ('posture','motion'):
            h=self.config['heads'][name]
            z=((flat.astype(np.float64)-np.asarray(h['mean'])[None,:,None])/np.asarray(h['std'])[None,:,None]).astype(np.float32)
            result=getattr(self,name)(torch.from_numpy(z)).numpy()
            if not np.isfinite(result).all(): raise ValueError('Nonfinite head logits')
            answer[name]=result
        return answer

    @torch.inference_mode()
    def infer(self, csi, *, subject, sequence, window, frames, stage7=False):
        identity = None
        try:
            identity = WindowID(subject, sequence, window, frames)
            x = preprocess_csi(csi, stage7=stage7)
        except (TypeError, ValueError, OverflowError) as error:
            return {
                'status': 'INVALID_INPUT', 'reason': str(error),
                'identity': asdict(identity) if identity else None,
                'candidate_state': None, 'state_name': None, 'posture': None, 'motion': None,
            }

        pose = self.root(torch.from_numpy(x)).numpy()
        logits = self.logits_from_pose(pose[None])
        posture_index, posture = head_output(logits['posture'][0], K4)
        motion_index, motion = head_output(logits['motion'][0], M3)
        candidate = state_code(posture_index, motion_index)
        return {
            'status': 'VALID', 'identity': asdict(identity),
            'posture': posture, 'motion': motion,
            'candidate_state': candidate, 'state_name': STATE_NAMES[candidate],
            'root_sha256': CHECKPOINTS['root'][0],
        }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/insense_v2.json'))
    for name in ('root','posture','motion'):
        parser.add_argument('--'+name,type=Path,default=Path(f'models/V2-{name}.pt'))
    for name in ('csi','identity'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage7',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    try:
        ids = json.loads(args.identity.read_text(encoding='utf-8-sig'))
        if not isinstance(ids, dict) or set(ids) != {'subject', 'sequence', 'window', 'frames'}:
            raise ValueError('Identity must contain subject, sequence, window and frames')
        csi = np.load(args.csi, allow_pickle=False)
        if not isinstance(csi, np.ndarray):
            csi.close()
            raise ValueError('Expected a single NPY CSI array')
    except (OSError, ValueError, EOFError) as error:
        parser.error(str(error))
    net = InSenseV2(args.config, root=args.root, posture=args.posture, motion=args.motion)
    result = net.infer(csi, stage7=args.stage7, **ids)
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'VALID' else 2


if __name__=='__main__':
    raise SystemExit(main())
