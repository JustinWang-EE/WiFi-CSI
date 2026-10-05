"""Evaluate a trusted CNN v0 checkpoint on the MM-Fi held-out cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wifi_densepose.data import CachedMMFiDataset
from wifi_densepose.metrics import pa_mpjpe_per_sample, raw_mpjpe, root_aligned_mpjpe
from wifi_densepose.models import SmallCSIPoseCNN
from wifi_densepose.train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Load only a checkpoint you trust")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dataset = CachedMMFiDataset(args.cache_dir, "validation")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SmallCSIPoseCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    squared_error = 0.0
    coordinate_count = 0
    raw_sum = 0.0
    root_sum = 0.0
    pa_sum = 0.0
    samples = 0
    for batch in loader:
        inputs = batch["input_wifi-csi"].to(device)
        targets = batch["output"].to(device)
        predictions = model(inputs)
        difference = predictions - targets
        count = predictions.shape[0]
        squared_error += float(difference.square().sum())
        coordinate_count += difference.numel()
        raw_sum += float(raw_mpjpe(predictions, targets)) * count
        root_sum += float(root_aligned_mpjpe(predictions, targets)) * count
        pa_sum += float(pa_mpjpe_per_sample(predictions, targets).sum())
        samples += count

    metrics = {
        "samples": samples,
        "mse_m2": squared_error / coordinate_count,
        "raw_mpjpe_m": raw_sum / samples,
        "root_aligned_mpjpe_m": root_sum / samples,
        "pa_mpjpe_m": pa_sum / samples,
    }
    metrics.update(
        {
            f"{key.removesuffix('_m')}_mm": value * 1000
            for key, value in list(metrics.items())
            if key.endswith("_mpjpe_m")
        }
    )
    rendered = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
