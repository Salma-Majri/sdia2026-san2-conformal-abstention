"""Pilote du bruit : trouve les 3 valeurs de sigma (leger, moyen, fort) sur val.

Ne lit JAMAIS test_id ni test_ood.
Sortie : affiche les sigmas a copier dans protocol.yaml.
"""
import copy
import json
import sys
import argparse
import numpy as np
import torch

from dataset import ISICDataset, build_transform, split_labels
from train import git_info, make_loader
from utils import (build_model, ensure_dir, get_device, load_config,
                   load_hf_dataset, load_split_indices, set_seed)

@torch.no_grad()
def get_logits_and_acc(model, loader, device):
    model.eval()
    out = []
    all_labels = []
    for x, y in loader:
        out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        all_labels.append(y.numpy())
    logits = np.concatenate(out).astype(np.float32)
    labels = np.concatenate(all_labels)
    acc = (logits.argmax(1) == labels).mean()
    return logits, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--seed", type=int, default=0, help="Doit etre 0 (voir protocole)")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed != 0:
        sys.exit("Le pilote du bruit doit etre fait avec la seed 0 (voir protocol.yaml).")

    set_seed(args.seed)
    device = get_device()
    
    ckpt = f"checkpoints/resnet50_main_seed{args.seed}.pt"
    hf = load_hf_dataset(cfg)
    splits = load_split_indices(cfg)
    val_idx = splits["val"]

    cfg_model = copy.deepcopy(cfg)
    model = build_model(cfg_model, pretrained=False)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    except FileNotFoundError:
        sys.exit(f"Checkpoint introuvable : {ckpt}.")
    model.to(device)

    sigma_grid = cfg["shift_simulation"]["acquisition_noise"]["pilot"]["sigma_grid"]
    target_drops = cfg["shift_simulation"]["acquisition_noise"]["target_accuracy_drop_points"]
    
    print("Calcul de l'accuracy de reference (sans bruit)...")
    ds_ref = ISICDataset(hf, val_idx, build_transform(cfg, train=False, noise_sigma=0.0))
    loader_ref = make_loader(ds_ref, 128, False, args.seed, args.num_workers)
    _, acc_ref = get_logits_and_acc(model, loader_ref, device)
    print(f"Accuracy de reference (val, seed {args.seed}) : {acc_ref:.4f}")

    results = {}
    for sigma in sigma_grid:
        print(f"Test sigma = {sigma}...")
        ds = ISICDataset(hf, val_idx, build_transform(cfg, train=False, noise_sigma=sigma))
        loader = make_loader(ds, 128, False, args.seed, args.num_workers)
        _, acc = get_logits_and_acc(model, loader, device)
        drop = (acc_ref - acc) * 100  # en points de pourcentage
        results[sigma] = {"acc": acc, "drop": drop}
        print(f"  -> Acc : {acc:.4f} | Chute : {drop:+.2f} points")

    # --- CORRECTION ICI : On force 3 sigmas distincts ---
    best_sigmas = {}
    used_sigmas = set()
    
    for level_name, target in target_drops.items():
        # On cherche le meilleur sigma parmi ceux qui ne sont pas encore utilisés
        available_sigmas = [s for s in results.keys() if s not in used_sigmas]
        best_sigma = min(available_sigmas, key=lambda s: abs(results[s]["drop"] - target))
        
        best_sigmas[level_name] = best_sigma
        used_sigmas.add(best_sigma)
        
        print(f"Cible '{level_name}' ({target} pts) -> sigma = {best_sigma} "
              f"(chute reelle : {results[best_sigma]['drop']:+.2f} pts)")
    # ---------------------------------------------------

    print("\n" + "="*60)
    print("RESULTAT : Copie ces lignes dans ton protocol.yaml")
    print("="*60)
    print("sigma_levels:")
    for k, v in best_sigmas.items():
        print(f"  {k}: {v}")
    print("="*60)

    out_dir = ensure_dir("outputs/results")
    out_path = out_dir / "pilot_noise_seed0.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "reference_acc": acc_ref,
            "results": {str(k): v for k, v in results.items()},
            "selected_sigmas": best_sigmas,
            "git_info": git_info()
        }, f, indent=2)
    print(f"\nResultats detailles sauvegardes dans {out_path}")

if __name__ == "__main__":
    main()