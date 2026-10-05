"""Train the reproducible CNN v0 baseline from a verified MM-Fi cache."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from wifi_densepose.data import CachedMMFiDataset
from wifi_densepose.models import SmallCSIPoseCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    squared_error = 0.0
    coordinate_count = 0
    joint_error = 0.0
    joint_count = 0
    sample_count = 0

    for batch in loader:
        inputs = batch["input_wifi-csi"].to(device)
        targets = batch["output"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(inputs)
        if not torch.isfinite(predictions).all():
            raise RuntimeError("non-finite model output")
        difference = predictions.detach() - targets
        squared_error += float(difference.square().sum())
        coordinate_count += difference.numel()
        joint_error += float(torch.linalg.vector_norm(difference, dim=-1).sum())
        joint_count += difference.shape[0] * difference.shape[1]
        sample_count += difference.shape[0]

    return {
        "mse": squared_error / coordinate_count,
        "raw_mpjpe": joint_error / joint_count,
        "samples": sample_count,
    }


def save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], epoch: int, metric: float) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"model_state_dict": model.state_dict(), "configuration": config, "epoch": epoch, "heldout_raw_mpjpe": metric},
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    device = choose_device(args.device)

    train_data = CachedMMFiDataset(args.cache_dir, "train")
    heldout_data = CachedMMFiDataset(args.cache_dir, "validation")
    expected = config["split"]
    if (len(train_data), len(heldout_data)) != (int(expected["train_samples"]), int(expected["heldout_samples"])):
        raise RuntimeError(f"unexpected split sizes: {len(train_data)}, {len(heldout_data)}")

    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, generator=generator, **config["train_loader"])
    heldout_loader = DataLoader(heldout_data, batch_size=batch_size, **config["heldout_loader"])

    model = SmallCSIPoseCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != int(config["model"]["expected_trainable_parameters"]):
        raise RuntimeError(f"unexpected trainable parameter count: {parameter_count}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_path = output / "epochs.csv"
    fieldnames = ["epoch", "train_mse", "train_raw_mpjpe", "heldout_mse", "heldout_raw_mpjpe", "new_best"]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, int(training["maximum_epochs"]) + 1):
            train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
            heldout_metrics = run_epoch(model, heldout_loader, loss_fn, device)
            metric = float(heldout_metrics["raw_mpjpe"])
            improved = metric < best_metric
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(output / "best.pt", model, config, epoch, metric)
            else:
                epochs_without_improvement += 1
            row = {
                "epoch": epoch,
                "train_mse": f"{float(train_metrics['mse']):.12f}",
                "train_raw_mpjpe": f"{float(train_metrics['raw_mpjpe']):.12f}",
                "heldout_mse": f"{float(heldout_metrics['mse']):.12f}",
                "heldout_raw_mpjpe": f"{metric:.12f}",
                "new_best": improved,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if epochs_without_improvement >= int(training["early_stopping_patience"]):
                break

    summary = {"best_epoch": best_epoch, "best_heldout_raw_mpjpe_m": best_metric, "device": str(device)}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

