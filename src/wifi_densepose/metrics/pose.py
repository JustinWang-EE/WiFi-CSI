"""Validated 3D pose metrics for MM-Fi evaluation."""

from __future__ import annotations

import torch


MMFI_ROOT_INDEX = 0


def _validate(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-2:] != (17, 3):
        raise ValueError(f"expected matching (N, 17, 3) poses, got {prediction.shape}, {target.shape}")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must use floating-point dtypes")


def raw_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute/world-coordinate MPJPE in the input coordinate unit."""
    _validate(prediction, target)
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()


def root_aligned_mpjpe(
    prediction: torch.Tensor,
    target: torch.Tensor,
    root_index: int = MMFI_ROOT_INDEX,
) -> torch.Tensor:
    """Translation-aligned MPJPE using the configured root joint."""
    _validate(prediction, target)
    if not 0 <= root_index < prediction.shape[1]:
        raise ValueError(f"root_index {root_index} is outside [0, {prediction.shape[1]})")
    pred_centered = prediction - prediction[:, root_index : root_index + 1]
    target_centered = target - target[:, root_index : root_index + 1]
    return torch.linalg.vector_norm(pred_centered - target_centered, dim=-1).mean()


def pa_mpjpe_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Similarity-align prediction to target using the MM-Fi row-vector convention."""
    _validate(prediction, target)
    target_mean = target.mean(dim=1, keepdim=True)
    prediction_mean = prediction.mean(dim=1, keepdim=True)
    target_zero = target - target_mean
    prediction_zero = prediction - prediction_mean
    target_norm = torch.linalg.vector_norm(target_zero, dim=(1, 2), keepdim=True)
    prediction_norm = torch.linalg.vector_norm(prediction_zero, dim=(1, 2), keepdim=True)
    if torch.any(target_norm == 0) or torch.any(prediction_norm == 0):
        raise ValueError("similarity alignment is undefined for zero-variance poses")

    target_unit = target_zero / target_norm
    prediction_unit = prediction_zero / prediction_norm
    covariance = target_unit.transpose(1, 2) @ prediction_unit
    u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    v = vh.transpose(1, 2)
    rotation = v @ u.transpose(1, 2)

    correction = torch.ones_like(singular_values)
    correction[:, -1] = torch.sign(torch.linalg.det(rotation))
    v = v * correction.unsqueeze(1)
    singular_values = singular_values * correction
    rotation = v @ u.transpose(1, 2)

    scale = singular_values.sum(dim=1, keepdim=True).unsqueeze(-1) * target_norm / prediction_norm
    translation = target_mean - scale * (prediction_mean @ rotation)
    aligned = scale * (prediction @ rotation) + translation
    return torch.linalg.vector_norm(aligned - target, dim=-1).mean(dim=1)


def pa_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Similarity/Procrustes-aligned MPJPE."""
    return pa_mpjpe_per_sample(prediction, target).mean()

