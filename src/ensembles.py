"""
ensembles.py - Moyenne des softmax des 3 seeds (sans réentraînement).
À partir des logits fournis dans outputs/logits/.
Sortie : outputs/ensembles/ensemble_{split}[_sigma{s}].npy
Note : PAS de seed dans le nom de fichier (car moyenne des 3 seeds).
Conforme aux specs de run_conformal.py (Salma).
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import ROOT

ENS_DIR = ROOT / "outputs" / "ensembles"
LOGITS_DIR = ROOT / "outputs" / "logits"

SPLITS = ["val", "calib", "ood_calib_pool", "test_id", "test_ood"]
SIGMA_LEVELS = [None, 0.01, 0.0125, 0.02]


def softmax(logits):
    """Softmax numérique stable."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def build_ensemble(split, sigma=None, seeds=(0, 1, 2)):
    """Moyenne des softmax des 3 seeds."""
    sigma_suffix = f"_sigma{sigma}" if sigma else ""
    
    all_probs = []
    for seed in seeds:
        logits_path = LOGITS_DIR / f"{split}_seed{seed}{sigma_suffix}.npy"
        if not logits_path.exists():
            # Fallback : essayer sans sigma pour les splits non bruités
            logits_path = LOGITS_DIR / f"{split}_seed{seed}.npy"
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits introuvables : {logits_path}")
        
        logits = np.load(logits_path).astype(np.float64)
        probs = softmax(logits)
        all_probs.append(probs)
    
    ensemble_probs = np.mean(all_probs, axis=0)
    
    # Vérifications strictes (comme dans run_conformal.py)
    assert ensemble_probs.shape[1] == 8, \
        f"Shape inattendu : {ensemble_probs.shape} (attendu [N, 8])"
    assert np.allclose(ensemble_probs.sum(axis=1), 1.0, atol=1e-3), \
        "Les probabilités de l'ensemble ne somment pas à 1"
    
    # Sauvegarde (PAS de seed dans le nom !)
    ENS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENS_DIR / f"ensemble_{split}{sigma_suffix}.npy"
    np.save(out_path, ensemble_probs)
    print(f" {out_path} | shape={ensemble_probs.shape} | sum_check={ensemble_probs[0].sum():.4f}")
    
    return ensemble_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_only", action="store_true",
                    help="Ne générer que val (pour développement)")
    args = ap.parse_args()
    
    splits = ["val"] if args.val_only else SPLITS
    
    for split in splits:
        for sigma in SIGMA_LEVELS:
            # Ne pas appliquer de bruit sur les splits autres que test_id
            if split != "test_id" and sigma is not None:
                continue
            try:
                build_ensemble(split, sigma)
            except FileNotFoundError as e:
                print(f" {e} (skip)")


if __name__ == "__main__":
    main()