"""Inference architectures for the selected, frozen InSense V3 checkpoints."""
from __future__ import annotations

import torch
from torch import nn

from .cnn import SmallCSIPoseCNN


class RootV3(SmallCSIPoseCNN):
    """Small-B CNN, retaining the original checkpoint's parameter names."""

    def __init__(self):
        super().__init__()
        for index, cin, cout in ((0, 3, 12), (2, 12, 24), (4, 24, 32)):
            self.features[index] = nn.Conv2d(cin, cout, 3, stride=1, padding=1)
        self.regressor[1] = nn.Linear(32 * 4 * 2, 128)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 114, 10) or not len(x):
            raise ValueError('Expected [N,3,114,10] CSI frames')
        return super().forward(x)


class ResidualBlockV3(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.first_norm = nn.GroupNorm(8, channels)
        self.second = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.second_norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(.20)

    def forward(self, x):
        h = self.dropout(nn.functional.gelu(self.first_norm(self.first(x))))
        h = self.dropout(nn.functional.gelu(self.second_norm(self.second(h))))
        return h + x


class HeadV3(nn.Module):
    """K4 Small-C or M3 Small-B; FC hidden width stays at 64."""

    def __init__(self, task):
        super().__init__()
        if task not in ('posture', 'motion'):
            raise ValueError('Expected posture or motion task')
        self.task = task
        channels, count = (48, 3) if task == 'posture' else (32, 4)
        self.input = nn.Conv1d(48, channels, 1)
        self.blocks = nn.Sequential(*(ResidualBlockV3(channels, 2 ** i) for i in range(count)))
        self.hidden = nn.Linear(channels * (2 if task == 'posture' else 3), 64)
        self.dropout = nn.Dropout(.20)
        self.output = nn.Linear(64, 4 if task == 'posture' else 3)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (48, 64) or not len(x):
            raise ValueError('Expected [N,48,64] pose features')
        h = self.blocks(self.input(x))
        pools = [h.mean(-1), h.amax(-1)]
        if self.task == 'motion':
            pools.append((h[:, :, 1:] - h[:, :, :-1]).abs().mean(-1))
        return self.output(self.dropout(nn.functional.gelu(self.hidden(torch.cat(pools, 1)))))
