"""
mc_dropout.py - Inférence MC-Dropout pour SAN-2.
- 20 passes, dropout actif, BatchNorm en eval.
- Astuce : features du backbone extraites UNE SEULE FOIS, puis 20 passes dropout+fc.
- Sortie : probabilités BRUTES [N, 8] au format .npy (somme = 1).
- Pas de temperature scaling (appliqué uniquement à 'single' dans run_conformal.py).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from utils import ROOT, set_seed, load_config, get_model, ISICDataset, build_transform

MC_DIR = ROOT / "outputs" / "mcdropout"
NUM_PASSES = 20


def enable_dropout(model):
    """Active uniquement les couches Dropout, laisse BatchNorm en eval."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def extract_backbone_features(model, dataloader, device):
    """
    Extrait les features du backbone UNE SEULE FOIS pour tout le dataset.
    Retourne (features, labels, indices).
    """
    model.eval()
    backbone = model.backbone
    
    all_features, all_labels, all_indices = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, indices = batch
            else:
                images, labels = batch
                indices = torch.arange(len(labels))
            
            images = images.to(device)
            features = backbone(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu().numpy())
            all_indices.append(indices.cpu().numpy())
    
    return (torch.cat(all_features), 
            np.concatenate(all_labels), 
            np.concatenate(all_indices))


def run_mc_passes(model, features, num_passes=NUM_PASSES, seed=None):
    """
    Fait N passes dropout+fc sur les features pré-extraites.
    Retourne les probabilités moyennes [N, K].
    """
    if seed is not None:
        set_seed(seed)
    
    fc = model.fc
    device = features.device
    
    all_probs = []
    chunk_size = 512
    
    for start in range(0, len(features), chunk_size):
        feat_chunk = features[start:start+chunk_size].to(device)
        chunk_probs = []
        
        for _ in range(num_passes):
            enable_dropout(model)
            with torch.no_grad():
                logits = fc(feat_chunk)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            chunk_probs.append(probs)
        
        mean_probs = np.mean(chunk_probs, axis=0)
        all_probs.append(mean_probs)
    
    return np.concatenate(all_probs, axis=0)


def generate_mcdropout(cfg, split, seed, sigma=None, num_passes=NUM_PASSES):
    """Génère les prédictions MC-Dropout pour un split donné."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(cfg, pretrained=False)
    checkpoint_path = ROOT / "checkpoints" / f"resnet50_main_seed{seed}.pt"
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    
    transform = build_transform(cfg, train=False)
    dataset = ISICDataset(cfg=cfg, split=split, transform=transform, 
                          noise_sigma=sigma, return_index=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    features, labels, indices = extract_backbone_features(model, dataloader, device)
    probs = run_mc_passes(model, features, num_passes, seed=seed)
    
    assert probs.shape == (len(labels), cfg["dataset"]["n_classes"]), \
        f"Shape inattendu : {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), \
        "Les probabilités ne somment pas à 1"
    
    labels_path = ROOT / "outputs" / "logits" / f"{split}_labels.npy"
    if labels_path.exists():
        ref_labels = np.load(labels_path)
        if len(ref_labels) == len(labels):
            assert np.array_equal(labels, ref_labels), \
                f"Ordre des labels différent pour {split} seed {seed} !"
            print(f" Ordre vérifié : {split} seed {seed}")
    
    MC_DIR.mkdir(parents=True, exist_ok=True)
    sigma_suffix = f"_sigma{sigma}" if sigma else ""
    out_path = MC_DIR / f"mcdropout_{split}_seed{seed}{sigma_suffix}.npy"
    np.save(out_path, probs)
    print(f" {out_path} | shape={probs.shape} | sum_check={probs[0].sum():.4f}")
    
    return probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num_passes", type=int, default=NUM_PASSES)
    ap.add_argument("--val_only", action="store_true",
                    help="Ne générer que val (pour développement)")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    splits = ["val"] if args.val_only else ["val", "calib", "ood_calib_pool", 
                                             "test_id", "test_ood"]
    sigmas = [None, 0.01, 0.0125, 0.02]
    
    for seed in args.seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        for split in splits:
            for sigma in sigmas:
                if split != "test_id" and sigma is not None:
                    continue
                generate_mcdropout(cfg, split, seed, sigma, args.num_passes)


if __name__ == "__main__":
    main()