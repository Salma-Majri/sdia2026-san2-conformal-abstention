"""Extrait les logits bruts (sans softmax) d'un checkpoint sur les splits du protocole.

Sorties (outputs/logits/) :
    {split}_seed{s}.npy     logits float32, shape [N, n_classes], meme ordre que splits[split]
    {split}_labels.npy      labels int64, shape [N]
    meta_seed{s}.json       checkpoint, commit git, formes

Usage :
    python src/extract_logits.py --seed 0
    python src/extract_logits.py --seed 0 --splits val --noise_sigma 0.05
"""
import argparse
import copy
import json
import sys

import numpy as np
import torch

from dataset import ISICDataset, build_transform, split_labels
from train import git_info, make_loader
from utils import (build_model, ensure_dir, get_device, load_config,
                   load_hf_dataset, load_split_indices, set_seed)

ALL_SPLITS = ["val", "calib", "test_id", "ood_calib_pool", "test_ood"]
NOISE_SPLITS = {"val", "test_id"}  

@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="defaut : checkpoints/resnet50_main_seed{seed}.pt")
    ap.add_argument("--splits", nargs="+", default=None, choices=ALL_SPLITS)
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0) # 0 pour Windows, 2 sur Colab
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed not in cfg["seeds"]["training"]:
        sys.exit(f"Seed {args.seed} absente de protocol.yaml {cfg['seeds']['training']}")

    if args.noise_sigma > 0:
        if args.splits is None:
            sys.exit("Avec --noise_sigma, precise --splits (val ou test_id uniquement).")
        bad = set(args.splits) - NOISE_SPLITS
        if bad:
            sys.exit(f"Bruit autorise seulement sur {sorted(NOISE_SPLITS)}, pas sur {sorted(bad)}.")
    splits_to_run = args.splits or ALL_SPLITS

    set_seed(args.seed)
    device = get_device()
    
    ckpt = args.ckpt or f"checkpoints/resnet50_main_seed{args.seed}.pt"

    hf = load_hf_dataset(cfg)
    splits = load_split_indices(cfg)

    cfg_model = copy.deepcopy(cfg)
    model = build_model(cfg_model, pretrained=False)
    
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    except FileNotFoundError:
        sys.exit(f"Checkpoint introuvable : {ckpt}. As-tu bien lance l'entrainement ?")
        
    model.to(device)

    out_dir = ensure_dir("outputs/logits")
    suffix = "" if args.noise_sigma <= 0 else f"_sigma{args.noise_sigma:g}"
    shapes = {}

    for name in splits_to_run:
        idx = splits[name]
        ds = ISICDataset(hf, idx, build_transform(cfg, train=False, noise_sigma=args.noise_sigma))
        loader = make_loader(ds, args.batch_size, False, args.seed, args.num_workers)
        
        logits = get_logits(model, loader, device)
        labels = split_labels(hf, idx).astype(np.int64)
        
        assert logits.shape == (len(idx), cfg["dataset"]["n_classes"])

        np.save(out_dir / f"{name}_seed{args.seed}{suffix}.npy", logits)
        
        labels_path = out_dir / f"{name}_labels.npy"
        if not labels_path.exists():
            np.save(labels_path, labels)
            
        shapes[name] = list(logits.shape)

        msg = f"{name:15s} logits {logits.shape}"
        if name == "val": 
            msg += f" | acc val = {(logits.argmax(1) == labels).mean():.4f}"
        print(msg)

    meta = {"seed": args.seed, "checkpoint": ckpt, "noise_sigma": args.noise_sigma,
            "dataset_revision": cfg["dataset"]["revision"], "shapes": shapes, **git_info()}
    with open(out_dir / f"meta_seed{args.seed}{suffix}.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nLogits ecrits dans {out_dir}")

if __name__ == "__main__":
    main()