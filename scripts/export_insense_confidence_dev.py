"""Reproduce the common-window CPU DEV audit from read-only project artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wifi_densepose.confidence import ConfidenceModel, WindowID, describe, combine, checkpoint_state

STARTS = (0,32,64,96,128,160,192,224,233)


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:raise ValueError(message)


def key(row):return '/'.join(row[k] for k in ('scene','subject','action'))


def run(a):
    torch.set_num_threads(4)
    net=ConfidenceModel(a.config,root=a.root,posture=a.posture,motion=a.motion)
    manifest=read(a.cache/'manifest.json');split=read(a.l3/'split_manifest.json')
    require(manifest['root_sha256']==net.config['checkpoints']['root'] and
            manifest['l3_split_sha256']==digest(a.l3/'split_manifest.json'),'Pose cache/split binding mismatch')
    require(manifest['root_sha256']==digest(a.root),'D0 identity mismatch')
    # Original labels and D0 window contents are checked against recorded digests.
    for name in ('d0_pose_sequences.npy','motion/dev_windows.npy','motion/dev_labels.npy','posture/dev_windows.npy'):
        sha=manifest['files'].get(name,manifest['files'].get(name.replace('/','\\')))
        require(digest(a.cache/name)==sha,f'Cache artifact changed: {name}')
    require(digest(a.l3/'dev_labels.npy')==manifest['motion_labels_source_sha256']['dev'],'Original labels changed')
    require(digest(a.posture_sequences)==manifest['action_metadata_sha256'],'Posture metadata changed')
    pose=np.load(a.cache/'d0_pose_sequences.npy',mmap_mode='r')
    poses_by_id={k:i for i,k in enumerate(manifest['sequence_ids'])}
    require(len(poses_by_id)==len(pose)==864 and pose.shape==(864,297,17,3) and
            manifest['sequence_ids']==[key(r) for r in split['sequences']], 'Sequence manifest mismatch')
    selection=[r for r in split['sequences'] if r['split']=='dev'];s_ids={key(r):r for r in selection}
    require(len(s_ids)==108 and set(split['dev'])=={r['subject'] for r in selection}
            and not set(split['dev'])&set(split['test']), 'Unexpected DEV split')
    p_state=checkpoint_state(a.posture,'posture');m_state=checkpoint_state(a.motion,'motion')
    mapping=p_state['taxonomy']['action27_to_class_index']
    p_ids={name:i for i,name in enumerate(p_state['dev_sequence_ids'])}
    with a.posture_sequences.open(encoding='utf-8-sig',newline='') as f:
        seq_list=[key(r) for r in csv.DictReader(f) if key(r) in p_ids]
    require(seq_list==p_state['dev_sequence_ids'] and len(seq_list)==216,'Posture DEV identities changed')
    with (a.l3/'window_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        win=sorted([r for r in csv.DictReader(f) if r['split']=='dev'],key=lambda r:int(r['slot']))
    expected=[(key(r),ws) for r in selection for ws in STARTS]
    require([(key(r),int(r['window_start'])) for r in win]==expected and [int(r['slot']) for r in win]==list(range(972)),
            'Motion frame/window order changed')
    y=np.array([int(r['label']) for r in win]);active=np.flatnonzero(y>=0)
    require(np.array_equal(y,np.load(a.l3/'dev_labels.npy')) and np.array_equal(y,np.load(a.cache/'motion/dev_labels.npy'))
            and active.tolist()==m_state['dev_window_slots'] and y[active].tolist()==m_state['dev_truth'], 'Motion target binding failed')
    p_norm=np.load(a.cache/'posture/dev_windows.npy',mmap_mode='r');m_norm=np.load(a.cache/'motion/dev_windows.npy',mmap_mode='r')
    records=[];motion_logits=[];posture_logits=[]
    for start in range(0,len(win),128):
        rows=win[start:start+128];indices=[int(r['slot']) for r in rows]
        raw=np.stack([pose[poses_by_id[key(r)],int(r['window_start']):int(r['window_start'])+64] for r in rows])
        flat=raw[:,:,1:,:].reshape(len(rows),64,48).transpose(0,2,1)
        old_post=[9*p_ids[key(r)]+STARTS.index(int(r['window_start'])) for r in rows]
        for head,reference in [('posture',p_norm[old_post]),('motion',m_norm[indices])]:
            n=net.config['heads'][head]
            expected=((flat.astype(np.float64)-np.array(n['mean'])[None,:,None])/np.array(n['std'])[None,:,None]).astype(np.float32)
            np.testing.assert_allclose(expected,reference,rtol=2e-6,atol=2e-6)
        scores=net.logits_from_pose(raw)
        motion_logits.extend(scores['motion']);posture_logits.extend(scores['posture'])
        for i,r in enumerate(rows):
            ws=int(r['window_start']);ident=WindowID(r['subject'],key(r),r['slot'],tuple(range(ws,ws+64)))
            p=describe('posture',scores['posture'][i],ident,net.config); m=describe('motion',scores['motion'][i],ident,net.config)
            records.append(dict(identity=asdict(ident),labels={'posture':mapping[r['action']],'motion':int(r['label']) if int(r['label'])>=0 else None},
                                posture=asdict(p),motion=asdict(m),policy_version=net.config['policy_version'],
                                root_sha256=net.config['checkpoints']['root'],**combine(p,m,net.config)))
    require(np.asarray(motion_logits)[active].argmax(1).tolist()==m_state['dev_predictions'],'Motion replay predictions changed')
    for i,row in enumerate(selection):
        a1=np.asarray(posture_logits)[9*i:9*(i+1)].mean(0)
        np.testing.assert_allclose(a1,p_state['dev_logits'][p_ids[key(row)]],atol=2e-4,rtol=2e-5)
    # Full Stage7 CSI -> D0 -> both classifier heads on the same observed frames.
    csi=np.load(a.stage7/'train_csi.npy',mmap_mode='r');metadata=np.load(a.stage7/'train_metadata.npy',mmap_mode='r')
    samples=[]
    for subject in split['dev']:
        r=next(r for r in win if r['subject']==subject);slot=int(r['slot']);ws=int(r['window_start']);s=s_ids[key(r)]['start']+ws
        require(s_ids[key(r)]['part']=='train','Unexpected cache partition')
        meta=metadata[s:s+64]
        require(np.array_equal(meta['idx'],np.arange(ws,ws+64)) and
                all(np.all(meta[field]==r[field]) for field in ('scene','subject','action')), 'CSI window IDs differ')
        result=net.infer(csi[s:s+64],stage7=True,**records[slot]['identity'])
        require(result['status']=='UNKNOWN' and result['identity']==records[slot]['identity'],'CPU smoke policy or IDs changed')
        with torch.inference_mode():
            direct_pose=net.root(torch.from_numpy(np.array(csi[s:s+64],copy=True))).numpy()
        cache_pose=pose[poses_by_id[key(r)],ws:ws+64]
        pose_diff=float(np.max(np.abs(direct_pose-cache_pose)))
        require(pose_diff <= 5e-6,'D0 CPU pose differs from cached values')
        logit_diff={}
        for head in ('posture','motion'):
            logit_diff[head]=float(np.max(np.abs(np.array(result[head]['logits'])-records[slot][head]['logits'])))
            require(logit_diff[head]<=.002 and result[head]['top1']==records[slot][head]['top1'], 'CPU replay changed logits or top class')
        samples.append(dict(subject=subject,status=result['status'],pose_max_abs_difference=pose_diff,logit_max_abs_difference=logit_diff))
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'common_dev_confidence.jsonl').open('x',encoding='utf-8') as f:
        for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
    (a.output/'cpu_smoke.json').write_text(json.dumps(dict(status='PASS',common_windows=972,motion_labeled=850,
        posture_sequence_replay=108,exact_motion_predictions=850,smoke_windows=samples,checkpoint_sha256=net.config['checkpoints']),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('PASS: 972 matched DEV windows, exact 850 Motion predictions, 108 Posture sequence logits, 4 full CPU windows')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','root','posture','motion','cache','l3','stage7','posture-sequences','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
