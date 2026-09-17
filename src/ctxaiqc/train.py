"""Training entry point.

One script, one configuration file, one seed. The checkpoint records the
configuration that produced it, so an experiment can always be traced back.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data import load_dataset
from .metrics import balanced_accuracy
from .models import build_model
from .utils import ensure_dir, load_config, save_json, seed_everything


def _loaders(data, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def make(x, y, shuffle):
        ds = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return make(data.x_train, data.y_train, True), make(data.x_val, data.y_val, False)


def train_model(cfg: dict) -> dict:
    """Train according to ``cfg`` and write a checkpoint. Returns the history."""
    import torch
    from torch import nn

    seed_everything(cfg["seed"])
    device = torch.device(cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = load_dataset(**cfg["dataset"], seed=cfg["seed"])
    train_loader, val_loader = _loaders(data, cfg["train"]["batch_size"])

    model_cfg = dict(cfg["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, n_classes=data.n_classes, **model_cfg).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_score, best_state, best_epoch = -np.inf, None, -1
    start = time.time()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += float(loss) * len(xb)
        scheduler.step()

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
                targets.append(yb.numpy())
        val_probs, val_y = np.concatenate(probs), np.concatenate(targets)
        val_score = balanced_accuracy(val_y, val_probs)

        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_loader.dataset),
                "val_balanced_accuracy": val_score,
                "lr": optimiser.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  "
            f"val balanced accuracy {val_score:.4f}"
        )

        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint_path = Path(cfg["train"]["checkpoint"])
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "state_dict": best_state,
            "model_name": model_name,
            "model_kwargs": {k: v for k, v in model_cfg.items() if k != "dropout"},
            "dropout": model_cfg.get("dropout", 0.3),
            "n_classes": data.n_classes,
            "class_names": data.class_names,
            "dataset": data.summary(),
            "config": cfg,
            "epoch": best_epoch,
            "val_balanced_accuracy": best_score,
        },
        checkpoint_path,
    )

    record = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_score,
        "seconds": time.time() - start,
        "device": str(device),
        "dataset": data.summary(),
    }
    save_json(record, Path(cfg["experiment"]["out_dir"]) / "training_history.json")
    print(f"best epoch {best_epoch} (val balanced accuracy {best_score:.4f}) -> {checkpoint_path}")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    train_model(load_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
