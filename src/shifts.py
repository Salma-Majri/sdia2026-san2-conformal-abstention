"""
shifts.py - Decalages controles evalues sur un split ID (protocol.yaml, shift_simulation).

Deux decalages, calcules a partir des MEMES probabilites et de la MEME calibration (calib ID) :
  - sous-population rare : couverture / taille / singletons par sous-groupe (centre, classe rare),
    evaluation conditionnelle, sans reechantillonnage ;
  - desequilibre de prevalence : reechantillonnage pondere AVEC remise (n_draws tirages) selon
    3 scenarios : uniform, inverse, rare_boost.
Aucun reentrainement, aucune recalibration. Le bruit d'acquisition est traite a part
(logits bruites extraits par extract_logits.py).

Le split evalue est val en mode smoke (test du code) et test_id en mode final.
Les centres des images viennent de la colonne `center` du dataset HF ; ils sont mis en cache
dans outputs/cache/centers.npy (une seule lecture du dataset).
"""
import itertools

import numpy as np

from conformal import calibrate, predict_sets
from utils import ROOT, ensure_dir, load_hf_dataset, load_split_indices

CENTERS_CACHE = ROOT / "outputs" / "cache" / "centers.npy"
SCENARIOS = ("uniform", "inverse", "rare_boost")
PREV_METRICS = ("coverage", "mean_set_size", "singleton_rate", "selective_risk")


# ----------------------------------------------------------------------
# Configuration (valeurs par defaut = protocole v1.1 si le YAML n'est pas detaille)
# ----------------------------------------------------------------------
def shift_cfg(cfg):
    sc = cfg.get("shift_simulation") or {}
    sub = sc.get("subpopulation") if isinstance(sc.get("subpopulation"), dict) else {}
    prev = sc.get("prevalence_shift") if isinstance(sc.get("prevalence_shift"), dict) else {}
    by_center = sub.get("by_center") if isinstance(sub.get("by_center"), dict) else {}
    by_class = sub.get("by_class") if isinstance(sub.get("by_class"), dict) else {}
    scen = prev.get("scenarios")
    scen = [s for s in scen if s in SCENARIOS] if isinstance(scen, dict) else []
    return {
        "rare_centers": list(by_center.get("rare_centers", [4, 5])),
        "rare_classes": list(by_class.get("rare_classes", [3, 5, 6, 7])),
        "min_n": int(sub.get("min_samples_report", 30)),
        "scenarios": scen or list(SCENARIOS),
        "n_draws": int(prev.get("n_draws", 200)),
        "draw_size": prev.get("draw_size"),
        "seed0": int(prev.get("resampling_seed", 0)),
        "boost": 5.0,
    }


# ----------------------------------------------------------------------
# Centres des images d'un split
# ----------------------------------------------------------------------
def load_centers(cfg, split):
    """Centre d'acquisition de chaque image du split, dans l'ordre de splits[split]."""
    if CENTERS_CACHE.exists():
        all_centers = np.load(CENTERS_CACHE)
    else:
        hf = load_hf_dataset(cfg)
        all_centers = np.array(hf["center"]).astype(np.int64)
        ensure_dir(CENTERS_CACHE.parent)
        np.save(CENTERS_CACHE, all_centers)
    return all_centers[load_split_indices(cfg)[split]]


# ----------------------------------------------------------------------
# Quantites par image (une fois par configuration de calibration)
# ----------------------------------------------------------------------
def _per_image(sets, y):
    sizes = sets.sum(1)
    hit = sets[np.arange(len(y)), y]
    single = sizes == 1
    err = single & (sets.argmax(1) != y)          # erreur parmi les ensembles de taille 1
    return sizes, hit, single, err


# ----------------------------------------------------------------------
# Sous-population rare
# ----------------------------------------------------------------------
def subpopulation_groups(y, centers, sc):
    groups = [("tous", np.ones(len(y), dtype=bool))]
    for c in sorted(set(centers.tolist())):
        groups.append((f"centre_{c}", centers == c))
    rare_c = [c for c in sc["rare_centers"] if c in set(centers.tolist())]
    if rare_c:
        groups.append(("centres_rares", np.isin(centers, rare_c)))
    for k in sc["rare_classes"]:
        groups.append((f"classe_{k}", y == k))
    groups.append(("classes_rares", np.isin(y, sc["rare_classes"])))
    return groups


# ----------------------------------------------------------------------
# Desequilibre de prevalence
# ----------------------------------------------------------------------
def target_prevalence(scenario, p_calib, rare_classes, boost):
    """Prevalence cible par classe.

    uniform    : 1/K pour chaque classe
    inverse    : proportionnelle a 1 / prevalence dans calib
    rare_boost : prevalence de calib, x boost sur les classes rares, puis renormalisation
    """
    K = len(p_calib)
    if scenario == "uniform":
        pi = np.ones(K)
    elif scenario == "inverse":
        pi = 1.0 / np.maximum(p_calib, 1e-12)
    elif scenario == "rare_boost":
        pi = p_calib * np.where(np.isin(np.arange(K), rare_classes), boost, 1.0)
    else:
        raise ValueError(f"scenario inconnu : {scenario}")
    return pi / pi.sum()


def _draw_metrics(idx, sizes, hit, single, err):
    ns = single[idx].sum()
    return {"coverage": hit[idx].mean(),
            "mean_set_size": sizes[idx].mean(),
            "singleton_rate": single[idx].mean(),
            "selective_risk": err[idx].sum() / ns if ns > 0 else np.nan}


# ----------------------------------------------------------------------
# Point d'entree : une methode, une seed
# ----------------------------------------------------------------------
def run_shifts(method, seed, probs, y, centers, cal_data, cfg, scores, conds, alpha, split):
    """Retourne (lignes sous-population, lignes prevalence). Calibration : source ID (calib)."""
    K = cfg["dataset"]["n_classes"]
    min_samples = cfg["conformal"]["class_conditional_min_samples"]
    sc = shift_cfg(cfg)
    cp, cy = cal_data
    p_calib = np.bincount(cy, minlength=K) / len(cy)
    n_c = np.bincount(y, minlength=K)
    draw_size = int(sc["draw_size"] or len(y))
    groups = subpopulation_groups(y, centers, sc)

    sub_rows, prev_rows = [], []
    for score, cond in itertools.product(scores, conds):
        cal = calibrate(cp, cy, alpha, score, cond, min_samples)
        sets = predict_sets(probs, cal)
        sizes, hit, single, err = _per_image(sets, y)
        base = {"method": method, "seed": seed, "split": split, "score": score,
                "conditioning": cond, "alpha": alpha}

        for name, mask in groups:
            n = int(mask.sum())
            if n == 0:
                continue
            sub_rows.append({**base, "group": name, "n": n, "indicatif": n < sc["min_n"],
                             "coverage": float(hit[mask].mean()),
                             "mean_set_size": float(sizes[mask].mean()),
                             "singleton_rate": float(single[mask].mean())})

        for scen in sc["scenarios"]:
            pi = target_prevalence(scen, p_calib, sc["rare_classes"], sc["boost"])
            w = pi[y] / np.maximum(n_c[y], 1)          # prob. par image = pi_classe / effectif_classe
            w = w / w.sum()
            vals = {m: [] for m in PREV_METRICS}
            for i in range(sc["n_draws"]):
                idx = np.random.default_rng(sc["seed0"] + i).choice(len(y), size=draw_size,
                                                                     replace=True, p=w)
                for m, v in _draw_metrics(idx, sizes, hit, single, err).items():
                    vals[m].append(v)
            row = {**base, "scenario": scen, "n_draws": sc["n_draws"], "draw_size": draw_size,
                   "target_prevalence": pi.tolist()}
            for m, v in vals.items():
                row[f"{m}_mean"] = float(np.nanmean(v))
                row[f"{m}_std"] = float(np.nanstd(v))
            prev_rows.append(row)
    return sub_rows, prev_rows