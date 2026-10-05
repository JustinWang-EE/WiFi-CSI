"""Build the lossless P3/S2 NumPy cache using a separate official MM-Fi toolbox checkout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


EXPECTED = {"train": 256_608, "validation": 64_152}
HELDOUT = ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]
TRAIN = [f"S{i:02d}" for i in range(1, 41) if f"S{i:02d}" not in HELDOUT]
METADATA_DTYPE = np.dtype([("scene", "<U3"), ("subject", "<U3"), ("action", "<U3"), ("idx", "<i2")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--toolbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def official_config() -> dict[str, object]:
    return {
        "modality": "wifi-csi",
        "protocol": "protocol3",
        "data_unit": "frame",
        "split_to_use": "cross_subject_split",
        "init_rand_seed": 0,
        "train_loader": {"batch_size": 64},
        "validation_loader": {"batch_size": 64},
        "cross_subject_split": {
            "train_dataset": {"split": "training", "scenes": None, "subjects": TRAIN, "actions": "all"},
            "val_dataset": {"split": "validation", "scenes": None, "subjects": HELDOUT, "actions": "all"},
        },
    }


def build_split(dataset: torch.utils.data.Dataset, split: str, output: Path, batch_size: int, make_dataloader) -> None:
    count = len(dataset)
    if count != EXPECTED[split]:
        raise RuntimeError(f"unexpected {split} sample count: {count}")
    final = {name: output / f"{split}_{name}.npy" for name in ("csi", "pose", "metadata")}
    if any(path.exists() for path in final.values()):
        raise FileExistsError(f"refusing to overwrite existing {split} cache")
    temporary = {name: path.with_suffix(".npy.building") for name, path in final.items()}
    csi = np.lib.format.open_memmap(temporary["csi"], mode="w+", dtype=np.float32, shape=(count, 3, 114, 10))
    pose = np.lib.format.open_memmap(temporary["pose"], mode="w+", dtype=np.float32, shape=(count, 17, 3))
    metadata = np.lib.format.open_memmap(temporary["metadata"], mode="w+", dtype=METADATA_DTYPE, shape=(count,))
    loader = make_dataloader(dataset, is_training=False, generator=torch.Generator().manual_seed(0), batch_size=batch_size)
    offset = 0
    for batch in loader:
        inputs = batch["input_wifi-csi"]
        targets = batch["output"]
        if tuple(inputs.shape[1:]) != (3, 114, 10) or tuple(targets.shape[1:]) != (17, 3):
            raise RuntimeError("unexpected official-loader tensor shape")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise RuntimeError("official loader returned non-finite values")
        end = offset + len(inputs)
        csi[offset:end] = inputs.numpy()
        pose[offset:end] = targets.numpy()
        for key in ("scene", "subject", "action", "idx"):
            metadata[key][offset:end] = batch[key]
        offset = end
    if offset != count:
        raise RuntimeError(f"wrote {offset} of {count} {split} samples")
    csi.flush(); pose.flush(); metadata.flush()
    del csi, pose, metadata
    for name in final:
        os.replace(temporary[name], final[name])


def main() -> None:
    args = parse_args()
    toolbox_root = args.toolbox_root.expanduser().resolve()
    sys.path.insert(0, str(toolbox_root))
    try:
        from mmfi_lib.mmfi import make_dataloader, make_dataset
    except ImportError as exc:
        raise RuntimeError("--toolbox-root must point to the official MMFi_dataset checkout") from exc

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_dataset, heldout_dataset = make_dataset(str(args.dataset_root.expanduser().resolve()), official_config())
    build_split(train_dataset, "train", output, args.batch_size, make_dataloader)
    build_split(heldout_dataset, "validation", output, args.batch_size, make_dataloader)
    manifest = {
        "protocol": "P3/S2 cross-subject",
        "train_samples": len(train_dataset),
        "heldout_samples": len(heldout_dataset),
        "heldout_subjects": HELDOUT,
        "format": "float32 NumPy arrays, loadable with mmap_mode='r'",
    }
    (output / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

