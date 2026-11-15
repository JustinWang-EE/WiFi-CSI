"""The release models and input/output contract on CPU."""

import hashlib
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from wifi_densepose.v2_inference import (
    CHECKPOINTS, K4, M3, STATE_NAMES, InSenseV2, checkpoint_state, head_output,
    load_config, preprocess_csi, state_code,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs' / 'insense_v2.json'
MODELS = {name: REPO / 'models' / f'V2-{name}.pt' for name in CHECKPOINTS}
IDENTITY = {'subject': 'example', 'sequence': 'sequence-1',
            'window': 'window-1', 'frames': list(range(64))}
CSI_SHAPE = (64, 3, 114, 10)


class InSenseV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = InSenseV2(CONFIG, **MODELS)

    def test_weights_and_strict_cpu_loading(self):
        counts = {}
        for name, path in MODELS.items():
            digest, n, _ = CHECKPOINTS[name]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            model = getattr(self.model, name)
            self.assertFalse(model.training)
            self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
            counts[name] = sum(p.numel() for p in model.parameters())
            self.assertEqual(counts[name], n)
        self.assertEqual(counts, {'root': 95827, 'posture': 111492, 'motion': 115523})

    def test_checkpoint_allowlist_rejects_modified_bytes(self):
        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'modified.pt'
            bad.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                checkpoint_state(bad, 'root')

    def test_config_rejects_invalid_class_order_and_normalization(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.json'
            cfg = load_config(CONFIG)
            cfg['heads']['posture']['classes'].reverse()
            path.write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'class order'):
                load_config(path)
            cfg = load_config(CONFIG)
            cfg['heads']['motion']['std'][0] = 0
            path.write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'normalization'):
                load_config(path)

    def test_raw_csi_normalization_and_missing_value_repair(self):
        raw = np.broadcast_to(np.arange(10, dtype=float), CSI_SHAPE).copy()
        raw[0, 0, 0, 3] = np.nan
        normalized = preprocess_csi(raw)
        expected = np.broadcast_to(np.arange(10, dtype=np.float32) / 9, CSI_SHAPE)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(normalized, expected)
        raw[0, :, :, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing CSI slice'):
            preprocess_csi(raw)

    def test_unscaled_probabilities_and_stable_top1(self):
        logits = np.array([0., 1., 2.])
        i, result = head_output(logits, M3)
        self.assertEqual((i, result['predicted_class']), (2, M3[2]))
        expected = np.exp(logits) / np.exp(logits).sum()
        np.testing.assert_allclose(list(result['class_probabilities_pct'].values()), 100 * expected)
        idx, result = head_output([1000., 1000., 1000., 1000.], K4)
        self.assertEqual(idx, 0)
        np.testing.assert_array_equal(list(result['class_probabilities_pct'].values()), [25.] * 4)

    def test_state_mapping(self):
        expected = ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S20', 'S21', 'S22']
        self.assertEqual([state_code(p, m) for p in range(3) for m in range(3)], expected)
        self.assertEqual([state_code(3, m) for m in range(3)], ['S30'] * 3)
        self.assertEqual(set(STATE_NAMES), set(expected + ['S30']))

    def test_missing_frames_and_invalid_csi(self):
        csi = np.zeros(CSI_SHAPE, dtype=np.float32)
        bad_values = csi.copy()
        bad_values[0, 0, 0, 0] = np.nan
        for data, ids in (
            (csi[:-1], IDENTITY),
            (csi, IDENTITY | {'frames': list(range(63))}),
            (csi, IDENTITY | {'frames': [*range(63), 64]}),
            (csi, IDENTITY | {'frames': list(reversed(range(64)))}),
            (csi, IDENTITY | {'frames': None}),
            (bad_values, IDENTITY),
            (csi + 2, IDENTITY),
            (csi.astype(np.complex64), IDENTITY),
        ):
            result = self.model.infer(data, stage7=True, **ids)
            self.assertEqual(result['status'], 'INVALID_INPUT')
            self.assertIsNone(result['candidate_state'])
            self.assertIsNone(result['posture'])
            self.assertIsNone(result['motion'])

    def test_end_to_end_cpu_inference(self):
        observed = {}
        hook = self.model.root.register_forward_hook(
            lambda model, args, pose: observed.update(csi=tuple(args[0].shape), pose=tuple(pose.shape))
        )
        csi = np.random.default_rng(0).random(CSI_SHAPE, dtype=np.float32)
        try:
            result = self.model.infer(csi, stage7=True, **IDENTITY)
        finally:
            hook.remove()
        self.assertEqual(observed, {'csi': CSI_SHAPE, 'pose': (64, 17, 3)})
        self.assertEqual(result['status'], 'VALID')
        self.assertEqual(result['identity']['frames'], tuple(IDENTITY['frames']))
        indices = []
        for head, classes in (('posture', K4), ('motion', M3)):
            output = result[head]
            scores = output['class_probabilities_pct']
            self.assertEqual(tuple(scores), classes)
            self.assertAlmostEqual(sum(scores.values()), 100.)
            self.assertEqual(max(scores, key=scores.get), output['predicted_class'])
            indices.append(classes.index(output['predicted_class']))
        self.assertEqual(result['candidate_state'], state_code(*indices))
        self.assertEqual(result['state_name'], STATE_NAMES[result['candidate_state']])
        self.assertFalse(any('score' in key or 'probability' in key for key in result))
