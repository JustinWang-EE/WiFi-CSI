from __future__ import annotations

import numpy as np
import torch

from wifi_densepose.metrics import pa_mpjpe, raw_mpjpe, root_aligned_mpjpe


def official_similarity_error(prediction: np.ndarray, target: np.ndarray) -> float:
    errors = []
    for pred, gt in zip(prediction, target):
        mu_gt = gt.mean(axis=0)
        mu_pred = pred.mean(axis=0)
        gt0 = gt - mu_gt
        pred0 = pred - mu_pred
        norm_gt = np.linalg.norm(gt0)
        norm_pred = np.linalg.norm(pred0)
        gt0 /= norm_gt
        pred0 /= norm_pred
        u, singular, vh = np.linalg.svd(gt0.T @ pred0)
        v = vh.T
        rotation = v @ u.T
        sign = np.sign(np.linalg.det(rotation))
        v[:, -1] *= sign
        singular[-1] *= sign
        rotation = v @ u.T
        scale = singular.sum() * norm_gt / norm_pred
        translation = mu_gt - scale * (mu_pred @ rotation)
        aligned = scale * (pred @ rotation) + translation
        errors.append(np.linalg.norm(aligned - gt, axis=1).mean())
    return float(np.mean(errors))


def test_perfect_prediction() -> None:
    target = torch.randn(4, 17, 3, dtype=torch.float64)
    assert raw_mpjpe(target, target).item() == 0.0
    assert root_aligned_mpjpe(target, target).item() == 0.0
    assert pa_mpjpe(target, target).item() < 1e-12


def test_translation_is_removed() -> None:
    target = torch.randn(3, 17, 3, dtype=torch.float64)
    prediction = target + torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    assert raw_mpjpe(prediction, target).item() > 0.0
    assert root_aligned_mpjpe(prediction, target).item() < 1e-12
    assert pa_mpjpe(prediction, target).item() < 1e-12


def test_corrected_pa_matches_official_convention() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(size=(5, 17, 3))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    prediction = 1.8 * (target @ rotation) + np.array([-3.0, 0.4, 2.0])
    expected = official_similarity_error(prediction, target)
    actual = pa_mpjpe(torch.from_numpy(prediction), torch.from_numpy(target)).item()
    assert abs(actual - expected) < 1e-12

