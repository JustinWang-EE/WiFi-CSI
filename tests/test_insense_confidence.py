from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from wifi_densepose.confidence import (CHECKPOINTS, K4, M3, ConfidenceModel, WindowID, checkpoint_state,
    combine, describe, load_config, preprocess_csi, probabilities, state_code)
from wifi_densepose.confidence_calibration import assess, fit_temperature, split_rows, threshold_for_coverage
from wifi_densepose.models.cnn import SmallCSIPoseCNN
from wifi_densepose.models.insense_heads import MotionOrPostureTCN

CONFIG=Path(__file__).resolve().parents[1]/'configs'/'insense_v2_confidence.json'
ASSERT=unittest.TestCase()


def identity(subject='S00',start=0):
    return WindowID(subject,subject+'/A01',str(start),tuple(range(start,start+64)))


def predictions(cfg,posture=1,motion=2):
    return [describe(head,4*np.eye(n)[k],identity(),cfg) for head,n,k in [('posture',4,posture),('motion',3,motion)]]


def test_probability_vectors_and_stable_ties():
    p=probabilities([[10000.,-10000.,9999.],[5.,5.,5.]])
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1),1)
    a=describe('posture',[1.,1.,0.,0.],identity(),load_config(CONFIG))
    assert a.index==0 and a.margin==0 and a.classes==K4
    assert describe('motion',[0,1,2],identity(),load_config(CONFIG)).classes==M3
    with ASSERT.assertRaises(ValueError):probabilities([0,np.nan])
    with ASSERT.assertRaises(ValueError):probabilities([0,1],0)


def test_state_mapping_and_policy():
    expected=['S00','S01','S02','S10','S11','S12','S20','S21','S22']
    assert [state_code(p,m) for p in range(3) for m in range(3)]==expected
    assert {state_code(3,m) for m in [None,0,1,2]}=={'S30'}
    cfg=load_config(CONFIG)
    assert combine(*predictions(cfg),cfg)=={'status':'UNKNOWN','state':None,'reason':'CONFIDENCE_UNVERIFIED'}
    cfg['calibration_status']='VERIFIED'
    p,m=predictions(cfg)
    cfg['heads']['posture']['thresholds']={'confidence':p.confidence,'margin':p.margin}
    cfg['heads']['motion']['thresholds']={'confidence':m.confidence,'margin':m.margin}
    assert combine(p,m,cfg)['state']=='S12'
    cfg['heads']['motion']['thresholds']['confidence']=np.nextafter(m.confidence,1.)
    assert combine(p,m,cfg)['reason']=='MOTION_ABSTAIN'
    jump,_=predictions(cfg,posture=3)
    assert combine(jump,m,cfg)['state']=='S30'
    misaligned=describe('motion',m.logits,identity(start=1),cfg)
    assert combine(jump,misaligned,cfg)['status']=='INVALID_INPUT'
    cfg['heads']['posture']['thresholds']['margin']=np.nextafter(jump.margin,1.)
    assert combine(jump,m,cfg)['reason']=='POSTURE_ABSTAIN'


def test_shapes_missing_frames_and_output():
    torch.set_num_threads(2)
    for name,model in [('root',SmallCSIPoseCNN()),('posture',MotionOrPostureTCN(False)),('motion',MotionOrPostureTCN(True))]:
        assert sum(p.numel() for p in model.parameters())==CHECKPOINTS[name][1]
    stub=ConfidenceModel.__new__(ConfidenceModel)
    stub.config=load_config(CONFIG)
    stub.root=SmallCSIPoseCNN().eval();stub.posture=MotionOrPostureTCN(False).eval();stub.motion=MotionOrPostureTCN(True).eval()
    x=np.zeros((64,3,114,10),np.float32); ids=asdict(identity())
    result=stub.infer(x,**ids,stage7=True)
    assert result['status']=='UNKNOWN' and result['identity']==ids
    for key,n in [('posture',4),('motion',3)]:
        assert len(result[key]['logits'])==n and np.isfinite(result[key]['logits']).all()
        np.testing.assert_allclose(sum(result[key]['probabilities']),1)
    missing=ids|{'frames':tuple(k for k in range(65) if k!=30)}
    assert stub.infer(x,**missing,stage7=True)['status']=='INVALID_INPUT'
    assert stub.infer(x[1:],**ids,stage7=True)['status']=='INVALID_INPUT'
    x[1,0,0,0]=np.nan
    assert stub.infer(x,**ids,stage7=True)['status']=='INVALID_INPUT'
    assert preprocess_csi(np.broadcast_to(np.arange(10),(64,3,114,10))).shape==(64,3,114,10)


def test_config_roundtrip_and_weight_allowlist():
    with TemporaryDirectory() as tmp:
        cfg=load_config(CONFIG);other=Path(tmp)/'config.json'
        other.write_text(json.dumps(cfg),encoding='utf-8'); loaded=load_config(other)
        assert [asdict(p) for p in predictions(loaded)]==[asdict(p) for p in predictions(cfg)]
        assert combine(*predictions(loaded),loaded)==combine(*predictions(cfg),cfg)
        cfg['heads']['posture']['classes'].reverse()
        other.write_text(json.dumps(cfg),encoding='utf-8')
        with ASSERT.assertRaisesRegex(ValueError,'class order'):load_config(other)
        with ASSERT.assertRaisesRegex(ValueError,'SHA256'):checkpoint_state(other,'root')


def test_calibration_fit_is_deterministic_and_subject_separated():
    cfg=load_config(CONFIG); rows=[]
    logits=np.array([[3.,0.],[2.,1.],[3.,0.],[1.,2.]])
    assert fit_temperature(logits,[0,1,0,1])==fit_temperature(logits,[0,1,0,1])
    for subj in ('S01','S02'):
        for j,truth in enumerate((0,1,0,1)):
            ids=identity(subj,64*j);row={'identity':asdict(ids),'labels':{'posture':truth,'motion':truth}}
            for h,n in [('posture',4),('motion',3)]:
                scores=np.arange(n)*.0;scores[j%3]=4
                row[h]=asdict(describe(h,scores,ids,cfg))
            rows.append(row)
    a=assess(rows,['S01'],['S02']); perturbed=deepcopy(rows)
    perturbed[-1]['motion']['logits']=(10,-1,0)
    b=assess(perturbed,['S01'],['S02'])
    for name in ('posture','motion'):
        assert a['heads'][name]['temperature_exploratory']==b['heads'][name]['temperature_exploratory']
        for variant in a['heads'][name]['coverage']:
            assert [r['threshold_fit_only'] for r in a['heads'][name]['coverage'][variant]]==[
                    r['threshold_fit_only'] for r in b['heads'][name]['coverage'][variant]]
    with ASSERT.assertRaisesRegex(ValueError,'disjoint'):split_rows(rows,['S01'],['S01','S02'])
    assert threshold_for_coverage(np.array([.6,.6,.9]),2/3)==.6


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items() if name.startswith('test_'))
