"""V3 frozen identities, normalization, ordering, and inference contract."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from wifi_densepose.v3_inference import (
    InSenseV3, CHECKPOINTS, K4, M3, MAPPING, JOINT_ORDER, load_config, checkpoint_state, state_code,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs/insense_v3.json'
IDS = dict(subject='synthetic', sequence='smoke', window='0', frames=list(range(64)))


@pytest.fixture(scope='module')
def net():
    torch.set_num_threads(2)
    return InSenseV3(CONFIG)


def test_strict_identity_counts_and_cpu_shapes(net):
    shapes = {'root': ((2, 3, 114, 10), (2, 17, 3)),
              'posture': ((2, 48, 64), (2, 4)), 'motion': ((2, 48, 64), (2, 3))}
    for name, (incoming, outgoing) in shapes.items():
        model = getattr(net, name)
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == CHECKPOINTS[name][1]
        assert model(torch.zeros(incoming)).shape == outgoing
    assert sum(x[1] for x in CHECKPOINTS.values()) == 133842


def test_joint_order_and_no_second_pelvis_subtraction(net):
    x = np.arange(2 * 64 * 17 * 3, dtype=np.float32).reshape(2, 64, 17, 3)
    for task in ('posture', 'motion'):
        h = net.config['heads'][task]
        explicit = np.stack([x[:, :, j, axis] for j in range(1, 17) for axis in range(3)], 1)
        expected = ((explicit.astype(np.float64) - np.array(h['mean'])[None, :, None])
                    / np.array(h['std'])[None, :, None]).astype(np.float32)
        np.testing.assert_array_equal(net.pose_features(x, task), expected)
        changed = x.copy()
        changed[:, :, 0] += 12345
        np.testing.assert_array_equal(net.pose_features(changed, task), expected)
    assert JOINT_ORDER[8:11] == ('thorax', 'neck', 'head')


def test_every_state_branch():
    assert MAPPING == [['S00', 'S01', 'S02'], ['S10', 'S11', 'S12'],
                       ['S20', 'S21', 'S22'], ['S30', 'S30', 'S30']]
    assert [state_code(3, m) for m in range(3)] == ['S30'] * 3


@pytest.mark.parametrize('change', ['v2_stats', 'class_order', 'root_hash', 'mapping', 'joint_order'])
def test_config_rejects_wrong_contract(change, tmp_path):
    cfg = load_config(CONFIG)
    if change == 'v2_stats':
        old = json.loads((REPO / 'configs/insense_v2.json').read_text())
        cfg['heads']['posture']['mean'] = old['heads']['posture']['mean']
    elif change == 'class_order':
        cfg['heads']['motion']['classes'].reverse()
    elif change == 'root_hash':
        cfg['checkpoints']['root'] = '0' * 64
    elif change == 'mapping':
        cfg['state_mapping'][3][1] = 'S31'
    else:
        cfg['joint_order'][1:3] = reversed(cfg['joint_order'][1:3])
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(p)


def test_hash_checked_before_deserialization(tmp_path):
    p = tmp_path / 'wrong.pt'
    p.write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError, match='SHA256'):
        checkpoint_state(p, 'posture')


def test_invalid_and_missing_frames_fail_safely(net):
    x = np.ones((64, 3, 114, 10), dtype=np.float32) * .5
    broken = x.copy()
    broken[2, 0, 0, 0] = np.nan
    for array, ids in [(x[:-1], IDS), (broken, IDS), (x + 2, IDS),
                       (x, IDS | {'frames': list(reversed(range(64)))}),
                       (x, IDS | {'frames': None})]:
        result = net.infer(array, stage7=True, **ids)
        assert result['status'] == 'INVALID_INPUT' and result['candidate_state'] is None
        assert result['posture'] is None and result['motion'] is None


def test_seven_softmax_percentages_and_top1_mapping(net):
    x = np.random.default_rng(0).random((64, 3, 114, 10), dtype=np.float32)
    result = net.infer(x, stage7=True, **IDS)
    assert result['status'] == 'VALID' and result['candidate_only']
    indices = []
    for task, classes in [('posture', K4), ('motion', M3)]:
        scores = result[task]['class_probabilities_pct']
        assert tuple(scores) == classes and sum(scores.values()) == pytest.approx(100)
        assert result[task]['predicted_class'] == max(scores, key=scores.get)
        indices.append(classes.index(result[task]['predicted_class']))
    assert result['candidate_state'] == state_code(*indices)


def test_relative_model_paths_after_package_copy(tmp_path):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'models').mkdir()
    shutil.copyfile(CONFIG, tmp_path / 'configs/insense_v3.json')
    for name in CHECKPOINTS:
        shutil.copyfile(REPO / f'models/V3-{name}.pt', tmp_path / f'models/V3-{name}.pt')
    relocated = InSenseV3(tmp_path / 'configs/insense_v3.json')
    assert relocated.config == load_config(CONFIG)


@pytest.mark.parametrize('name,budget', [('root', 10), ('posture', 30), ('motion', 30)])
def test_released_training_log_budget_and_checkpoint_selection(name, budget):
    log = json.loads((REPO / f'training_logs/v3/{name}.json').read_text())
    rows = log['epochs']
    assert log['epochs_completed'] == len(rows) == budget
    assert [r['epoch'] for r in rows] == list(range(1, budget + 1))
    assert log['checkpoint']['sha256'] == CHECKPOINTS[name][0]
    assert log['checkpoint']['parameters'] == CHECKPOINTS[name][1]
    if name == 'root':
        best = min(rows, key=lambda r: (r['dev_root17_mpjpe_mm'], r['epoch']))
    elif name == 'posture':
        best = max(rows, key=lambda r: (r['dev_macro_f1'], r['dev_accuracy'], -r['dev_ce'], -r['epoch']))
    else:
        best = max(rows, key=lambda r: (r['dev_macro_f1'], -r['epoch']))
    assert log['best'] == best
    assert best['epoch'] == log['checkpoint']['best_epoch'] == CHECKPOINTS[name][2]
    assert all(np.isfinite(value) for row in rows for value in row.values())
