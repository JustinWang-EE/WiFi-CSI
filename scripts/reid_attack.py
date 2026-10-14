"""Re-identification (re-ID) attacker: how much subject identity leaks? (checklist B1 / B7)

Protocol (from the proposal):
  * Each MM-Fi environment has 10 subjects and every subject appears in exactly one
    environment, so we evaluate each environment separately (10-way, chance = 10%).
    A cross-environment classifier could "recognise the room" instead of the person.
  * Train / test split BY ACTION (default: test = A21-A27). Frames of one action are
    highly correlated in time; a random frame split would let the attacker memorise
    neighbouring frames and overstate leakage.
  * The official train and validation caches are merged so all 10 subjects of an
    environment are present.

Attacker inputs (``--inputs``):
  csi        raw CSI (3, 114, 10)                     -> fresh small CNN
  gt_pose    ground-truth skeleton, root-centred (51) -> MLP   (reference upper bound for skeleton leakage)
  pred_pose  pose-model output, root-centred (51)     -> MLP   (needs --checkpoint)
  feature    pose-model 512-d embedding               -> MLP   (needs --checkpoint)

The attacker is always trained from scratch with a fixed number of epochs and is
never selected on the test actions. For B7, point --checkpoint at a frozen DANN model.

Example:
  uv run python scripts/reid_attack.py --cache-dir D:\\Datasets\\mmfi-p3-s2-cache `
      --checkpoint outputs/baseline/best.pt --output outputs/reid/baseline.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wifi_densepose.models import SmallCSIPoseCNN
from wifi_densepose.models.dann import embed, pose_from_embedding
from wifi_densepose.train import choose_device, seed_everything

ALL_INPUTS = ("csi", "gt_pose", "pred_pose", "feature")
DEFAULT_TEST_ACTIONS = [f"A{i:02d}" for i in range(21, 28)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="pose model (baseline or DANN); needed for pred_pose/feature")
    parser.add_argument("--inputs", nargs="+", choices=ALL_INPUTS, help="default: all available")
    parser.add_argument("--test-actions", nargs="+", default=DEFAULT_TEST_ACTIONS)
    parser.add_argument("--environments", nargs="+", help="default: every environment in the cache")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


# ----------------------------------------------------------------------------- data
class MergedCache:
    """Official train + validation caches viewed as one array (CSI stays memory-mapped)."""

    def __init__(self, cache_dir: Path) -> None:
        self.csi_parts: list[tuple[int, np.ndarray]] = []
        poses, metas = [], []
        offset = 0
        for split in ("train", "validation"):
            csi = np.load(cache_dir / f"{split}_csi.npy", mmap_mode="r")
            pose = np.load(cache_dir / f"{split}_pose.npy")
            meta = np.load(cache_dir / f"{split}_metadata.npy")
            if not (len(csi) == len(pose) == len(meta)):
                raise ValueError(f"inconsistent {split} cache")
            self.csi_parts.append((offset, csi))
            poses.append(pose)
            metas.append(meta)
            offset += len(csi)
        self.pose = np.concatenate(poses)
        meta = np.concatenate(metas)
        self.scene = meta["scene"].astype(str)
        self.subject = meta["subject"].astype(str)
        self.action = meta["action"].astype(str)

    def read_csi(self, indices: np.ndarray) -> np.ndarray:
        """Rows ``indices`` (sorted, global) from the memory-mapped CSI files."""
        chunks = []
        for offset, csi in self.csi_parts:
            local = indices[(indices >= offset) & (indices < offset + len(csi))] - offset
            if len(local):
                chunks.append(np.asarray(csi[local], dtype=np.float32))
        return np.concatenate(chunks)


def root_centre(pose: torch.Tensor) -> torch.Tensor:
    return (pose - pose[:, 0:1]).reshape(len(pose), -1)


@torch.no_grad()
def run_pose_model(model: SmallCSIPoseCNN, csi: torch.Tensor, batch: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    features, poses = [], []
    for start in range(0, len(csi), batch):
        z = embed(model, csi[start : start + batch])
        features.append(z)
        poses.append(pose_from_embedding(model, z))
    return torch.cat(features), torch.cat(poses)


# ----------------------------------------------------------------------------- attackers
class CSIAttacker(nn.Module):
    """Same conv capacity as the pose backbone, with a 10-way subject head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPAttacker(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch]).argmax(1) for i in range(0, len(x), batch)])


def accuracy_report(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> dict[str, float]:
    recalls = [float((pred[y == c] == c).float().mean()) for c in range(num_classes) if (y == c).any()]
    return {"accuracy": float((pred == y).float().mean()), "balanced_accuracy": float(np.mean(recalls))}


def train_attacker(
    kind: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    seed_everything(args.seed)
    # Standardise with TRAIN statistics only.
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = CSIAttacker(num_classes) if kind == "csi" else MLPAttacker(x_train.shape[1], num_classes)
    model = model.to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)

    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(x_train.device)
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x_train[idx]), y_train[idx]).backward()
            optimizer.step()

    test = accuracy_report(predict(model, x_test), y_test, num_classes)
    train = accuracy_report(predict(model, x_train), y_train, num_classes)
    return {
        "test_accuracy": test["accuracy"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "train_accuracy": train["accuracy"],
    }


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    inputs = args.inputs or (list(ALL_INPUTS) if args.checkpoint else ["csi", "gt_pose"])
    if {"pred_pose", "feature"} & set(inputs) and args.checkpoint is None:
        raise SystemExit("--checkpoint is required for pred_pose / feature")

    pose_model = None
    if args.checkpoint is not None:
        pose_model = SmallCSIPoseCNN().to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        pose_model.load_state_dict(checkpoint["model_state_dict"])
        pose_model.eval()

    data = MergedCache(args.cache_dir)
    environments = args.environments or sorted(set(data.scene.tolist()))
    is_test_action = np.isin(data.action, args.test_actions)
    results: dict[str, dict] = {}

    for env in environments:
        start_time = time.perf_counter()
        rows = np.flatnonzero(data.scene == env)
        env_subjects = sorted(set(data.subject[rows].tolist()))
        label_of = {s: i for i, s in enumerate(env_subjects)}
        labels = torch.tensor([label_of[s] for s in data.subject[rows]], device=device)
        test_mask = torch.from_numpy(is_test_action[rows]).to(device)
        train_mask = ~test_mask

        csi = torch.from_numpy(data.read_csi(rows)).to(device)
        gt_pose = torch.from_numpy(data.pose[rows]).to(device)
        features = {"csi": csi, "gt_pose": root_centre(gt_pose)}
        if pose_model is not None:
            z, pred = run_pose_model(pose_model, csi)
            features["feature"] = z
            features["pred_pose"] = root_centre(pred)

        env_result: dict[str, object] = {
            "subjects": env_subjects,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "chance": 1.0 / len(env_subjects),
        }
        for kind in inputs:
            x = features[kind]
            env_result[kind] = train_attacker(
                kind, x[train_mask], labels[train_mask], x[test_mask], labels[test_mask], len(env_subjects), args
            )
            print(f"{env} {kind:9s} test_acc={env_result[kind]['test_accuracy']:.3f} "
                  f"(train_acc={env_result[kind]['train_accuracy']:.3f})", flush=True)
        env_result["seconds"] = round(time.perf_counter() - start_time, 1)
        results[env] = env_result
        del csi, features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean = {kind: float(np.mean([results[e][kind]["test_accuracy"] for e in environments])) for kind in inputs}
    report = {
        "protocol": {
            "per_environment": True,
            "split": "by_action",
            "test_actions": args.test_actions,
            "epochs": args.epochs,
            "seed": args.seed,
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        },
        "environments": results,
        "mean_test_accuracy": mean,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = "| env | " + " | ".join(inputs) + " |"
    print("\n" + header + "\n|" + "---|" * (len(inputs) + 1))
    for env in environments:
        print(f"| {env} | " + " | ".join(f"{100 * results[env][k]['test_accuracy']:.1f}%" for k in inputs) + " |")
    print("| mean | " + " | ".join(f"{100 * mean[k]:.1f}%" for k in inputs) + " |")


if __name__ == "__main__":
    main()
