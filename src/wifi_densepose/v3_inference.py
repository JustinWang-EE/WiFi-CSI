"""InSense V3: frozen Small-B Root, K4 Small-C, and M3 Small-B."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from .models.insense_v3 import RootV3, HeadV3
from .v2_inference import (
    K4, M3, K4_SHA, M3_SHA, STATE_NAMES, WINDOW, WindowID,
    head_output, preprocess_csi, state_code,
)

CHECKPOINTS = {
    'root': ('c42c0047f95ae7a631581b9b95bdb6aedcd9fb08a36ab63f544679d448ee3e75', 49371, 2),
    'posture': ('6971a90e05403b62e8324faa82af1ab0c05bb10b598b1124633f9a587b20b83c', 51156, 25),
    'motion': ('1188bfffe910395179a85f4abc161c5e937ff297bc76cbc25eb034623b4a881b', 33315, 26),
}
NORMALIZATION = {
    'posture': '2f9e598b3f1dc4d0deb85e0c3069cba0c13e923b61c3544671dd5f392a9c4449',
    'motion': 'c5e3bb992c1b459dd4a6b0767cc2ca6a020f8a774bea5501439432e3f4e4c679',
}
ARCHITECTURES = {
    'root': {'channels': [12, 24, 32], 'kernel': 3, 'stride': 1, 'padding': 1,
             'activation': 'ReLU', 'adaptive_pool': [4, 2], 'linear': [256, 128, 51]},
    'posture': {'channels': 48, 'blocks': 3, 'dilations': [1, 2, 4], 'kernel': 3,
                'groupnorm_groups': 8, 'activation': 'GELU', 'dropout': .20,
                'residual': True, 'pooling': ['mean', 'max'], 'linear': [96, 64, 4]},
    'motion': {'channels': 32, 'blocks': 4, 'dilations': [1, 2, 4, 8], 'kernel': 3,
               'groupnorm_groups': 8, 'activation': 'GELU', 'dropout': .20,
               'residual': True, 'pooling': ['mean', 'max', 'mean_abs_hidden_difference'],
               'linear': [96, 64, 3]},
}
MAPPING = [[state_code(p, m) for m in range(3)] for p in range(4)]
JOINT_ORDER = ('pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
               'left_ankle', 'spine', 'thorax', 'neck', 'head', 'left_shoulder',
               'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist')
INTERFACES = {'root': {'input': ['N', 3, 114, 10], 'output': ['N', 17, 3]},
              'posture': {'input': ['N', 48, 64], 'output': ['N', 4]},
              'motion': {'input': ['N', 48, 64], 'output': ['N', 3]}}


def normalization_digest(h):
    encoded = json.dumps({k: h[k] for k in ('mean', 'std')}, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg.get('schema') != 'insense-v3/1':
        raise ValueError('Unsupported V3 configuration')
    if cfg.get('checkpoints') != {name: terms[0] for name, terms in CHECKPOINTS.items()}:
        raise ValueError('V3 checkpoint allowlist mismatch')
    if cfg.get('architectures') != ARCHITECTURES or cfg.get('state_mapping') != MAPPING:
        raise ValueError('Wrong V3 architecture or state mapping')
    if tuple(cfg.get('joint_order', ())) != JOINT_ORDER or cfg.get('interfaces') != INTERFACES:
        raise ValueError('Wrong V3 joint order or tensor interfaces')
    if cfg.get('probabilities') != {'kind': 'unscaled_softmax', 'temperature': 1.0}:
        raise ValueError('V3 uses unscaled softmax only')
    for name, classes, taxonomy in [('posture', K4, K4_SHA), ('motion', M3, M3_SHA)]:
        h = cfg['heads'][name]
        if tuple(h['classes']) != classes or h['taxonomy_sha256'] != taxonomy:
            raise ValueError('Wrong class order or taxonomy')
        for key in ('mean', 'std'):
            v = np.asarray(h[key], dtype=np.float64)
            if v.shape != (48,) or not np.isfinite(v).all() or (key == 'std' and np.any(v <= 0)):
                raise ValueError('Invalid TRAIN normalization')
        if (h['root_sha256'] != CHECKPOINTS['root'][0]
                or normalization_digest(h) != NORMALIZATION[name]):
            raise ValueError('Wrong Small-B TRAIN normalization')
    paths = cfg.get('model_files', {})
    if set(paths) != set(CHECKPOINTS):
        raise ValueError('Missing V3 model files')
    for name, value in paths.items():
        if value != f'models/V3-{name}.pt':
            raise ValueError('Expected relative V3 model paths')
    return cfg


def checkpoint_state(path, name):
    payload = Path(path).read_bytes()
    digest, _, epoch = CHECKPOINTS[name]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'{name} V3 checkpoint SHA256 mismatch')
    # Only deserialize the exact hash-verified project checkpoint.
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    if state['epoch'] != epoch:
        raise ValueError('V3 checkpoint epoch mismatch')
    c = state['config']
    if name == 'root':
        if c['channels'] != [12, 24, 32] or c['parameters'] != 49371:
            raise ValueError('Expected Small-B Root')
    else:
        expected = ARCHITECTURES[name]
        if (c['root_sha256'] != CHECKPOINTS['root'][0]
                or c['taxonomy_sha256'] != (K4_SHA if name == 'posture' else M3_SHA)
                or any(c[k] != expected[k] for k in ('channels', 'blocks', 'dilations', 'groupnorm_groups'))
                or c['final_hidden_units'] != 64
                or normalization_digest(state['normalization']) != NORMALIZATION[name]):
            raise ValueError('V3 Root, head or normalization binding mismatch')
        if name == 'motion' and tuple(c['classes']) != M3:
            raise ValueError('Wrong Motion class order')
    return state


class InSenseV3:
    def __init__(self, config_path):
        path = Path(config_path).resolve()
        self.config = load_config(path)
        package = path.parent.parent
        self.root = RootV3()
        self.posture, self.motion = HeadV3('posture'), HeadV3('motion')
        for name in CHECKPOINTS:
            state = checkpoint_state(package / self.config['model_files'][name], name)
            model = getattr(self, name)
            model.load_state_dict(state['model'], strict=True)
            if sum(p.numel() for p in model.parameters()) != CHECKPOINTS[name][1]:
                raise ValueError('V3 parameter count mismatch')
            if name != 'root':
                for key in ('mean', 'std'):
                    if not np.array_equal(state['normalization'][key], self.config['heads'][name][key]):
                        raise ValueError('Config and checkpoint normalization differ')
            model.eval().requires_grad_(False)

    def pose_features(self, poses, name):
        x = np.asarray(poses)
        if (name not in ('posture', 'motion') or x.ndim != 4 or x.shape[1:] != (64, 17, 3)
                or not len(x) or not np.isfinite(x).all()):
            raise ValueError('Expected finite [N,64,17,3] predicted poses')
        # The Root is already root-relative. Drop joint 0 without another subtraction.
        flat = x[:, :, 1:, :].reshape(len(x), 64, 48).transpose(0, 2, 1)
        h = self.config['heads'][name]
        return ((flat.astype(np.float64) - np.asarray(h['mean'])[None, :, None])
                / np.asarray(h['std'])[None, :, None]).astype(np.float32)

    @torch.inference_mode()
    def logits_from_pose(self, poses):
        answer = {}
        for name in ('posture', 'motion'):
            model = getattr(self, name)
            x = torch.from_numpy(self.pose_features(poses, name)).to(next(model.parameters()).device)
            logits = model(x).cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError('Nonfinite head logits')
            answer[name] = logits
        return answer

    @torch.inference_mode()
    def infer(self, csi, *, subject, sequence, window, frames, stage7=False):
        identity = None
        try:
            identity = WindowID(subject, sequence, window, frames)
            x = preprocess_csi(csi, stage7=stage7)
        except (TypeError, ValueError, OverflowError) as error:
            return {'status': 'INVALID_INPUT', 'reason': str(error), 'candidate_state': None,
                    'posture': None, 'motion': None, 'identity': asdict(identity) if identity else None}
        pose = self.root(torch.from_numpy(x).to(next(self.root.parameters()).device)).cpu().numpy()
        if pose.shape != (64, 17, 3) or not np.isfinite(pose).all():
            raise ValueError('Invalid Root output')
        logits = self.logits_from_pose(pose[None])
        p, posture = head_output(logits['posture'][0], K4)
        m, motion = head_output(logits['motion'][0], M3)
        code = state_code(p, m)
        return {'status': 'VALID', 'candidate_only': True, 'identity': asdict(identity),
                'posture': posture, 'motion': motion, 'candidate_state': code,
                'state_name': STATE_NAMES[code], 'root_sha256': CHECKPOINTS['root'][0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'configs/insense_v3.json')
    parser.add_argument('--csi', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--stage7', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    ids = json.loads(args.identity.read_text(encoding='utf-8-sig'))
    if not isinstance(ids, dict) or set(ids) != {'subject', 'sequence', 'window', 'frames'}:
        parser.error('Identity requires subject, sequence, window and frames')
    x = np.load(args.csi, allow_pickle=False)
    if not isinstance(x, np.ndarray):
        x.close()
        parser.error('Expected one NPY CSI array')
    result = InSenseV3(args.config).infer(x, stage7=args.stage7, **ids)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
    print(encoded)
    return 0 if result['status'] == 'VALID' else 2


if __name__ == '__main__':
    raise SystemExit(main())
