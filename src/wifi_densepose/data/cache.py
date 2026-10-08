"""Memory-mapped, lossless cache access for official MM-Fi loader outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


METADATA_DTYPE = np.dtype(
    [
        ("scene", "<U3"),
        ("subject", "<U3"),
        ("action", "<U3"),
        ("idx", "<i2"),
    ]
)


class CachedMMFiDataset(Dataset):
    """Read exact float32 MM-Fi model tensors from memory-mapped `.npy` files."""

    def __init__(self, cache_directory: Path | str, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        cache_directory = Path(cache_directory)
        self.split = split
        self.csi = np.load(cache_directory / f"{split}_csi.npy", mmap_mode="r")
        self.pose = np.load(cache_directory / f"{split}_pose.npy", mmap_mode="r")
        self.metadata = np.load(
            cache_directory / f"{split}_metadata.npy", mmap_mode="r"
        )

        if self.csi.dtype != np.float32 or self.csi.shape[1:] != (3, 114, 10):
            raise ValueError(f"Invalid cached CSI array: {self.csi.shape}, {self.csi.dtype}")
        if self.pose.dtype != np.float32 or self.pose.shape[1:] != (17, 3):
            raise ValueError(f"Invalid cached pose array: {self.pose.shape}, {self.pose.dtype}")
        if not (len(self.csi) == len(self.pose) == len(self.metadata)):
            raise ValueError("Cached arrays have inconsistent sample counts")

    def __len__(self) -> int:
        return len(self.csi)

    def __getitem__(self, index: int) -> dict[str, object]:
        metadata = self.metadata[index]
        # Copy one small sample out of the read-only mmap before exposing it to PyTorch.
        return {
            "input_wifi-csi": torch.from_numpy(np.array(self.csi[index], copy=True)),
            "output": torch.from_numpy(np.array(self.pose[index], copy=True)),
            "scene": str(metadata["scene"]),
            "subject": str(metadata["subject"]),
            "action": str(metadata["action"]),
            "idx": int(metadata["idx"]),
        }
