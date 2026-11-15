from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from wifi_densepose.confidence import (CHECKPOINTS, K4, M3, ConfidenceModel, WindowID, checkpoint_state,
    combine, describe, load_config, preprocess_csi, probabilities, state_code)
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
    assert combine(*predictions(cfg),cfg)=={
        'status':'UNCERTAIN','state':None,'candidate_state':'S12','state_name':'Lower-body Transition',
        'reason':'CONFIDENCE_UNVERIFIED',
    }
    cfg['calibration_status']='VERIFIED'
    p,m=predictions(cfg)
    cfg['heads']['posture']['thresholds']={'confidence':p.confidence,'margin':p.margin}
    cfg['heads']['motion']['thresholds']={'confidence':m.confidence,'margin':m.margin}
    assert combine(p,m,cfg)['state']=='S12'
    cfg['heads']['motion']['thresholds']['confidence']=np.nextafter(m.confidence,1.)
    assert combine(p,m,cfg)['reason']=='MOTION_ABSTAIN'
    jump,_=predictions(cfg,posture=3)
    assert combine(jump,m,cfg)['candidate_state']=='S30'
    assert combine(jump,m,cfg)['status']=='UNCERTAIN'
    cfg['heads']['motion']['thresholds']['confidence']=m.confidence
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
    assert result['status']=='UNCERTAIN' and result['identity']==ids
    for key,n in [('posture',4),('motion',3)]:
        scores = result[key]['class_probabilities_pct']
        assert tuple(scores) == (K4 if key == 'posture' else M3)
        assert len(scores) == n and np.isfinite(list(scores.values())).all()
        np.testing.assert_allclose(sum(scores.values()),100)
        assert result[key]['confidence_pct'] == scores[result[key]['predicted_class']] == max(scores.values())
    assert result['candidate_state'] is not None and result['state'] is None
    assert not any('score' in field or 'probabilit' in field for field in result)
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
        cfg['heads']['motion']['thresholds']={'confidence':0.7,'margin':0.0}
        other.write_text(json.dumps(cfg),encoding='utf-8')
        with ASSERT.assertRaisesRegex(ValueError,'Unverified'):load_config(other)
        cfg['heads']['motion']['thresholds']=None
        cfg['heads']['posture']['classes'].reverse()
        other.write_text(json.dumps(cfg),encoding='utf-8')
        with ASSERT.assertRaisesRegex(ValueError,'class order'):load_config(other)
        with ASSERT.assertRaisesRegex(ValueError,'SHA256'):checkpoint_state(other,'root')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items() if name.startswith('test_'))
