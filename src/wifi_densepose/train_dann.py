"""Train the root-relative CNN v0, optionally with DANN identity-adversarial training.

One script covers checklist B2, B4, B5 and B6:

* B2  root-relative baseline         : ``--disable-adversary``
* B5  sanity check (must equal B2)    : ``--lambda 0``  (adversary on, but lambda = 0)
* B4/B6 DANN / lambda sweep          : ``--lambda 0.01`` ... ``--lambda 1``

Methodology differences from ``train.py``:

* Checkpoint selection and early stopping use a *validation* split carved out of the
  official training subjects (``split.validation_subjects``). The 8 official held-out
  subjects are touched exactly once, after training, to report the final numbers.
* The pose target is root-relative (``pose - pose[:, 0:1]``) by default.
* Checkpoints are only eligible after ``training.selection_start_epoch`` so that a
  DANN run cannot "win" by selecting an epoch where lambda had not ramped up yet.

The saved ``model_state_dict`` contains only the pose network, so the checkpoint
loads into ``SmallCSIPoseCNN`` / ``evaluate.py`` / ``scripts/reid_attack.py`` directly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from wifi_densepose.data import CachedMMFiDataset
from wifi_densepose.metrics import pa_mpjpe_per_sample, root_aligned_mpjpe
from wifi_densepose.models import SmallCSIPoseCNN
from wifi_densepose.models.dann import (
    IdentityHead,
    embed,
    ganin_lambda,
    grad_reverse,
    pose_from_embedding,
)
from wifi_densepose.train import choose_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dann.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lambda", dest="lambda_max", type=float, help="override adversary.lambda_max")
    parser.add_argument("--disable-adversary", action="store_true", help="plain root-relative baseline (B2)")
    parser.add_argument("--seed", type=int, help="override training.seed")
    return parser.parse_args()


def make_target(pose: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "root_relative":
        return pose - pose[:, 0:1]
    if mode == "absolute":
        return pose
    raise ValueError(f"unknown target mode: {mode}")


@torch.no_grad()
def evaluate_pose(model: SmallCSIPoseCNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Root-aligned and PA-MPJPE in metres. Both are invariant to the target mode."""
    model.eval()
    root_sum = 0.0
    pa_sum = 0.0
    samples = 0
    for batch in loader:
        inputs = batch["input_wifi-csi"].to(device)
        targets = batch["output"].to(device)
        predictions = model(inputs)
        count = predictions.shape[0]
        root_sum += float(root_aligned_mpjpe(predictions, targets)) * count
        pa_sum += float(pa_mpjpe_per_sample(predictions, targets).sum())
        samples += count
    return {"samples": samples, "root_aligned_mpjpe_m": root_sum / samples, "pa_mpjpe_m": pa_sum / samples}


def train_one_epoch(
    model: SmallCSIPoseCNN,
    head: IdentityHead | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    head_optimizer: torch.optim.Optimizer | None,
    subject_to_id: dict[str, int],
    target_mode: str,
    lambda_max: float,
    ramp_steps: int,
    step: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    model.train()
    if head is not None:
        head.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    pose_loss_sum = 0.0
    id_loss_sum = 0.0
    id_correct = 0
    samples = 0
    lam = 0.0

    for batch in loader:
        inputs = batch["input_wifi-csi"].to(device)
        targets = make_target(batch["output"].to(device), target_mode)

        optimizer.zero_grad(set_to_none=True)
        z = embed(model, inputs)                       # (N, 512)
        predictions = pose_from_embedding(model, z)    # (N, 17, 3)
        pose_loss = mse(predictions, targets)
        loss = pose_loss

        if head is not None:
            head_optimizer.zero_grad(set_to_none=True)
            lam = ganin_lambda(step / ramp_steps, lambda_max) if ramp_steps > 0 else lambda_max
            subject_ids = torch.tensor([subject_to_id[s] for s in batch["subject"]], device=device)
            logits = head(grad_reverse(z, lam))
            id_loss = ce(logits, subject_ids)
            # Head minimises id_loss; through the GRL the backbone receives -lam * d(id_loss)/dz.
            loss = loss + id_loss
            id_loss_sum += float(id_loss) * inputs.shape[0]
            id_correct += int((logits.argmax(1) == subject_ids).sum())

        loss.backward()
        optimizer.step()
        if head_optimizer is not None:
            head_optimizer.step()

        if not torch.isfinite(predictions).all():
            raise RuntimeError("non-finite model output")
        pose_loss_sum += float(pose_loss) * inputs.shape[0]
        samples += inputs.shape[0]
        step += 1

    metrics = {"train_pose_mse": pose_loss_sum / samples, "lambda": lam}
    if head is not None:
        metrics["train_id_loss"] = id_loss_sum / samples
        metrics["train_id_accuracy"] = id_correct / samples
    return metrics, step


def save_checkpoint(path: Path, model: nn.Module, head: nn.Module | None, extra: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "identity_head_state_dict": None if head is None else head.state_dict(),
            **extra,
        },
        temporary,
    )
    temporary.replace(path)


def to_mm(metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    out.update({k.removesuffix("_m") + "_mm": v * 1000 for k, v in metrics.items() if k.endswith("_mpjpe_m")})
    return out


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    training = config["training"]
    adversary = config["adversary"]
    if args.seed is not None:
        training["seed"] = args.seed
    if args.lambda_max is not None:
        adversary["lambda_max"] = args.lambda_max
    if args.disable_adversary:
        adversary["enabled"] = False

    seed = int(training["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    target_mode = config.get("target", "root_relative")

    # ---------------- data: official train -> (train, validation); official held-out -> test
    train_full = CachedMMFiDataset(args.cache_dir, "train")
    test_data = CachedMMFiDataset(args.cache_dir, "validation")
    split = config["split"]
    if "train_samples" in split and (len(train_full), len(test_data)) != (
        int(split["train_samples"]),
        int(split["heldout_samples"]),
    ):
        raise RuntimeError(f"unexpected split sizes: {len(train_full)}, {len(test_data)}")

    subjects = np.asarray(train_full.metadata["subject"]).astype(str)
    validation_subjects = list(split["validation_subjects"])
    missing = set(validation_subjects) - set(subjects)
    if missing:
        raise ValueError(f"validation subjects not in the official training split: {sorted(missing)}")
    is_validation = np.isin(subjects, validation_subjects)
    train_indices = np.flatnonzero(~is_validation).tolist()
    validation_indices = np.flatnonzero(is_validation).tolist()
    train_subjects = sorted(set(subjects[~is_validation].tolist()))
    subject_to_id = {s: i for i, s in enumerate(train_subjects)}

    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_full, train_indices), batch_size=batch_size, generator=generator, **config["train_loader"]
    )
    validation_loader = DataLoader(Subset(train_full, validation_indices), batch_size=batch_size, **config["eval_loader"])
    test_loader = DataLoader(test_data, batch_size=batch_size, **config["eval_loader"])

    # ---------------- models: pose net first, then head (keeps pose init identical with/without head)
    model = SmallCSIPoseCNN().to(device)
    pose_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if pose_parameters != int(config["model"]["expected_trainable_parameters"]):
        raise RuntimeError(f"unexpected pose-network parameter count: {pose_parameters}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    head = None
    head_optimizer = None
    lambda_max = float(adversary["lambda_max"]) if adversary["enabled"] else 0.0
    if adversary["enabled"]:
        head = IdentityHead(len(train_subjects), hidden=int(adversary["hidden"])).to(device)
        head_optimizer = torch.optim.Adam(head.parameters(), lr=float(adversary["learning_rate"]))

    ramp_steps = int(adversary["ramp_epochs"]) * len(train_loader)
    selection_start = int(training["selection_start_epoch"])
    patience = int(training["early_stopping_patience"])

    # ---------------- output
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "resolved": {
            "train_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_data),
            "device": str(device),
        },
    }
    (output / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(json.dumps(resolved["resolved"]), flush=True)

    fieldnames = [
        "epoch", "lambda", "train_pose_mse", "train_id_loss", "train_id_accuracy",
        "val_root_aligned_mpjpe", "val_pa_mpjpe", "eligible", "new_best", "seconds",
    ]
    best_metric = float("inf")
    best_epoch = 0
    stale = 0
    step = 0
    checkpoint_extra = {"configuration": resolved, "subject_to_id": subject_to_id, "target": target_mode}

    with (output / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, int(training["maximum_epochs"]) + 1):
            start = time.perf_counter()
            train_metrics, step = train_one_epoch(
                model, head, train_loader, optimizer, head_optimizer, subject_to_id,
                target_mode, lambda_max, ramp_steps, step, device,
            )
            val = evaluate_pose(model, validation_loader, device)
            metric = val["root_aligned_mpjpe_m"]
            eligible = epoch >= selection_start
            improved = eligible and metric < best_metric
            if improved:
                best_metric, best_epoch, stale = metric, epoch, 0
                save_checkpoint(output / "best.pt", model, head, {**checkpoint_extra, "epoch": epoch, "val_root_aligned_mpjpe_m": metric})
            elif eligible:
                stale += 1
            row = {
                "epoch": epoch,
                "lambda": f"{train_metrics['lambda']:.6f}",
                "train_pose_mse": f"{train_metrics['train_pose_mse']:.8f}",
                "train_id_loss": f"{train_metrics.get('train_id_loss', float('nan')):.6f}",
                "train_id_accuracy": f"{train_metrics.get('train_id_accuracy', float('nan')):.6f}",
                "val_root_aligned_mpjpe": f"{metric:.8f}",
                "val_pa_mpjpe": f"{val['pa_mpjpe_m']:.8f}",
                "eligible": eligible,
                "new_best": improved,
                "seconds": f"{time.perf_counter() - start:.1f}",
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            if eligible and stale >= patience:
                break

    save_checkpoint(output / "last.pt", model, head, {**checkpoint_extra, "epoch": epoch})
    if best_epoch == 0:
        raise RuntimeError("no eligible epoch: maximum_epochs < selection_start_epoch")

    # ---------------- final report: load best, touch the held-out test subjects once
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = {
        "best_epoch": best_epoch,
        "lambda_max": lambda_max,
        "adversary_enabled": bool(adversary["enabled"]),
        "target": target_mode,
        "seed": seed,
        "validation": to_mm(evaluate_pose(model, validation_loader, device)),
        "test": to_mm(evaluate_pose(model, test_loader, device)),
    }
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
