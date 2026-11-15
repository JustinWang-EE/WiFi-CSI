"""Checkpoint-compatible K4 and M3 heads for InSense V2."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.first = nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation)
        self.first_norm = nn.GroupNorm(8, 64)
        self.second = nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation)
        self.second_norm = nn.GroupNorm(8, 64)
        self.dropout = nn.Dropout(.20)

    def forward(self, x):
        h = self.dropout(F.gelu(self.first_norm(self.first(x))))
        h = self.dropout(F.gelu(self.second_norm(self.second(h))))
        return x + h


class MotionOrPostureTCN(nn.Module):
    def __init__(self, motion):
        super().__init__()
        self.motion = motion
        self.input = nn.Conv1d(48, 64, 1)
        self.blocks = nn.Sequential(*(ResidualTemporalBlock(d) for d in (1, 2, 4, 8)))
        self.hidden = nn.Linear(192 if motion else 128, 64)
        self.dropout = nn.Dropout(.20)
        self.output = nn.Linear(64, 3 if motion else 4)

    def forward(self, x):
        h = self.blocks(self.input(x))
        features = [h.mean(-1), h.amax(-1)]
        if self.motion:
            features.append((h[:,:,1:] - h[:,:,:-1]).abs().mean(-1))
        return self.output(self.dropout(F.gelu(self.hidden(torch.cat(features, 1)))))
