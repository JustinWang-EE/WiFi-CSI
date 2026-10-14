from __future__ import annotations

import math

import torch

from wifi_densepose.models import SmallCSIPoseCNN
from wifi_densepose.models.dann import (
    IdentityHead,
    embed,
    ganin_lambda,
    grad_reverse,
    pose_from_embedding,
)


def test_grad_reverse_is_identity_forward_and_negated_backward() -> None:
    x = torch.randn(5, 7, requires_grad=True)
    y = grad_reverse(x, 0.3)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.3))


def test_ganin_schedule_endpoints() -> None:
    assert ganin_lambda(0.0, 0.5) == 0.0
    assert math.isclose(ganin_lambda(1.0, 0.5), 0.5 * (2 / (1 + math.exp(-10)) - 1))
    assert ganin_lambda(2.0, 0.5) == ganin_lambda(1.0, 0.5)  # clamped
    values = [ganin_lambda(p / 10, 1.0) for p in range(11)]
    assert values == sorted(values)  # monotone


def test_split_forward_matches_model_forward() -> None:
    torch.manual_seed(0)
    model = SmallCSIPoseCNN().eval()
    x = torch.randn(4, 3, 114, 10)
    z = embed(model, x)
    assert z.shape == (4, 512)
    assert torch.equal(pose_from_embedding(model, z), model(x))


def test_lambda_zero_leaves_backbone_gradient_unchanged() -> None:
    torch.manual_seed(0)
    model = SmallCSIPoseCNN()
    head = IdentityHead(28)
    x = torch.randn(8, 3, 114, 10)
    target = torch.randn(8, 17, 3)
    ids = torch.randint(0, 28, (8,))

    z = embed(model, x)
    torch.nn.functional.mse_loss(pose_from_embedding(model, z), target).backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    z = embed(model, x)
    loss = torch.nn.functional.mse_loss(pose_from_embedding(model, z), target)
    loss = loss + torch.nn.functional.cross_entropy(head(grad_reverse(z, 0.0)), ids)
    loss.backward()
    for p, g in zip(model.parameters(), reference):
        assert torch.equal(p.grad, g)
    assert any(p.grad.abs().sum() > 0 for p in head.parameters())  # head still learns
