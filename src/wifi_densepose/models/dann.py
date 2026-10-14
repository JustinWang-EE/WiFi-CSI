"""DANN building blocks: gradient reversal, Ganin lambda schedule, identity head.

The pose network itself is the unchanged ``SmallCSIPoseCNN``. We only split its
forward pass at the 512-d ``Flatten`` output so an identity head can be attached
there. Because the pose network is untouched, a DANN checkpoint's
``model_state_dict`` loads into ``SmallCSIPoseCNN`` (and ``evaluate.py``) as-is.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from wifi_densepose.models.cnn import SmallCSIPoseCNN

FEATURE_DIM = 64 * 4 * 2  # 512


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass, multiplies the gradient by ``-lam`` in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return GradReverse.apply(x, lam)


def ganin_lambda(progress: float, lambda_max: float) -> float:
    """Ganin & Lempitsky (2015) schedule: 0 at progress=0, -> lambda_max as progress -> 1."""
    p = min(max(progress, 0.0), 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def embed(model: SmallCSIPoseCNN, wifi_csi: torch.Tensor) -> torch.Tensor:
    """CSI -> 512-d feature z (conv stack + Flatten)."""
    return model.regressor[0](model.features(wifi_csi))


def pose_from_embedding(model: SmallCSIPoseCNN, z: torch.Tensor) -> torch.Tensor:
    """512-d feature z -> (N, 17, 3) pose (the remaining MLP layers)."""
    return model.regressor[1:](z).reshape(-1, 17, 3)


class IdentityHead(nn.Module):
    """Subject classifier attached through the GRL. Training-only; dropped at deployment."""

    def __init__(self, num_subjects: int, hidden: int = 128, in_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_subjects),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
