"""Entrainement ResNet50 sur le split `train`, early stopping sur `val`.

Usage :
    python src/train.py --seed 0
    python src/train.py --seed 0 --smoke        # test rapide (petits sous-ensembles, 1 epoch)
"""
import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import ISICDataset, build_transform, split_labels
from utils import (ROOT, build_model, ensure_dir, get_device, load_config,
                   load_hf_dataset, load_split_indices, set_seed)


def git_info():
    """Commit courant et etat du depot (propre ou non), pour la reproductibilite."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}


def run_metadata(cfg, args, device):
    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_revision": cfg["dataset"]["revision"],
        "smoke": args.smoke,
        "num_workers": args.num_workers,
        **git_info(),
    }


def worker_init(worker_id):
    # PyTorch donne deja a chaque worker une graine derivee du generator du loader
    # (differente a chaque epoque). On n'ecrase que le RNG numpy, pour ne pas figer
    # les augmentations d'une epoque a l'autre.
    np.random.seed(torch.initial_seed() % 2 ** 32)


def make_loader(ds, batch_size, shuffle, seed, num_workers):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=g, worker_init_fn=worker_init)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n, preds, ys = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total += criterion(out, y).item() * len(y)
        n += len(y)
        preds.append(out.argmax(1).cpu().numpy())
        ys.append(y.cpu().numpy())
    preds, ys = np.concatenate(preds), np.concatenate(ys)
    return {
        "loss": total / n,
        "acc": float((preds == ys).mean()),
        "macro_f1": float(f1_score(ys, preds, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="test rapide, pas pour les resultats")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed not in cfg["seeds"]["training"]:
        sys.exit(f"Seed {args.seed} absente de protocol.yaml {cfg['seeds']['training']} : "
                 "ajoute-la au protocole (commit dedie) avant de lancer.")
    set_seed(args.seed)
    device = get_device()
    tc = cfg["training"]

    hf = load_hf_dataset(cfg)
    splits = load_split_indices(cfg)
    train_idx, val_idx = splits["train"], splits["val"]
    if args.smoke:
        rng = np.random.RandomState(0)
        train_idx = rng.choice(train_idx, 256, replace=False)
        val_idx = rng.choice(val_idx, 128, replace=False)

    train_ds = ISICDataset(hf, train_idx, build_transform(cfg, train=True))
    val_ds = ISICDataset(hf, val_idx, build_transform(cfg, train=False))
    train_loader = make_loader(train_ds, tc["batch_size"], True, args.seed, args.num_workers)
    val_loader = make_loader(val_ds, tc["batch_size"], False, args.seed, args.num_workers)

    model = build_model(cfg).to(device)

    weight = None
    if tc["class_weighting"]:
        counts = np.bincount(split_labels(hf, train_idx), minlength=cfg["dataset"]["n_classes"])
        w = counts.sum() / (len(counts) * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    max_epochs = 1 if args.smoke else tc["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    use_amp = bool(tc["mixed_precision"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = ensure_dir("checkpoints")
    res_dir = ensure_dir("outputs/results")
    tag = "smoke" if args.smoke else "main"
    ckpt_path = ckpt_dir / f"resnet50_{tag}_seed{args.seed}.pt"

    best_val, bad_epochs, history = float("inf"), 0, []
    patience = tc["early_stopping"]["patience"]

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(y)
            n += len(y)
        scheduler.step()

        val = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": run_loss / n, **{f"val_{k}": v for k, v in val.items()},
               "time_s": round(time.time() - t0, 1)}
        history.append(row)
        print(row)

        if val["loss"] < best_val:
            best_val, bad_epochs = val["loss"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping a l'epoch {epoch}")
                break

    out = res_dir / f"train_{tag}_seed{args.seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "best_val_loss": best_val,
                   "meta": run_metadata(cfg, args, device), "config": cfg,
                   "history": history}, f, indent=2)
    print(f"Checkpoint : {ckpt_path}\nHistorique : {out}")


if __name__ == "__main__":
    main()