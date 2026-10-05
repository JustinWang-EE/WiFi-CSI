"""Compact WiFi CSI to 3D pose baseline."""

from __future__ import annotations

import torch
from torch import nn


class SmallCSIPoseCNN(nn.Module):
    """CNN v0: (3, 114, 10) CSI input to (17, 3) pose output."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 17 * 3),
        )

    def forward(self, wifi_csi: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(wifi_csi)).reshape(-1, 17, 3)

