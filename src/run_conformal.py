"""
run_conformal.py - Lance la prediction conforme sur les logits (ou probabilites) du protocole.

Deux modes :
  --mode smoke   calibration sur `calib`, evaluation sur `val`. Verifie le pipeline.
                 AUCUN jeu de test n'est lu. Peut etre relance autant de fois que necessaire.
  --mode final   calibration (calib / ood_calib_pool / mixte), evaluation sur test_id,
                 test_id bruite (sigma_levels du protocole) et test_ood.
                 UNE SEULE EXECUTION par fichier de sortie : refuse de tourner si le fichier
                 existe deja, si l'arbre git a des modifications non commitees, ou si
                 sigma_levels est null.

Methodes (--methods, defaut : single) :
  single     un modele par seed : logits bruts / T (temperature ajustee sur val)
  mcdropout  outputs/mcdropout/mcdropout_{split}_seed{N}[_sigma{s}].npy  (probabilites)
  ensemble   outputs/ensembles/ensemble_{split}[_sigma{s}].npy           (probabilites)
Les probabilites de mcdropout et ensemble sont utilisees telles quelles (pas de temperature).

Ablations :
  --no_ts            sans temperature scaling (single uniquement)
  --n_ood 50 100 361 taille du pool OOD pour les sources ood_few_shot et mixed

Usage (depuis n'importe quel dossier) :
    python src/run_conformal.py --mode smoke
    python src/run_conformal.py --mode final
    python src/run_conformal.py --mode final --no_ts
    python src/run_conformal.py --mode final --n_ood 50 100 361
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from conformal import (LOGITS_DIR, build_calibration_set, calibrate, load_probs,
                       load_temperature, predict_sets)
from metrics import evaluate_probs, evaluate_sets
from utils import ROOT, ensure_dir, load_config

MC_DIR = ROOT / "outputs" / "mcdropout"
ENS_DIR = ROOT / "outputs" / "ensembles"


# ----------------------------------------------------------------------
# Git (garde-fou du mode final)
# ----------------------------------------------------------------------
def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def git_state():
    """(commit, dirty) ; dirty ignore les fichiers non suivis (les sorties ne comptent pas)."""
    try:
        return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


# ----------------------------------------------------------------------
# Chargement des probabilites
# ----------------------------------------------------------------------
def _sig(sigma):
    return f"_sigma{sigma:g}" if sigma else ""


def load_method(method, split, seed, use_ts, sigma=None):
    """Retourne (probs [N, K] float64, labels [N])."""
    if method == "single":
        T = load_temperature(seed) if use_ts else 1.0
        return load_probs(split, seed, T, sigma)
    labels = np.load(LOGITS_DIR / f"{split}_labels.npy").astype(np.int64)
    if method == "mcdropout":
        path = MC_DIR / f"mcdropout_{split}_seed{seed}{_sig(sigma)}.npy"
    elif method == "ensemble":
        path = ENS_DIR / f"ensemble_{split}{_sig(sigma)}.npy"
    else:
        raise ValueError(f"methode inconnue : {method}")
    if not path.exists():
        sys.exit(f"Fichier introuvable : {path}")
    p = np.load(path).astype(np.float64)
    if p.shape[0] != len(labels):
        sys.exit(f"{path.name} : {p.shape[0]} lignes, {len(labels)} labels attendus")
    if not np.allclose(p.sum(1), 1.0, atol=1e-3):
        sys.exit(f"{path.name} : les lignes ne somment pas a 1 (probabilites attendues, pas des logits)")
    return p, labels


def sigma_values(cfg):
    lv = cfg["shift_simulation"]["acquisition_noise"].get("sigma_levels")
    if not lv:
        return None
    return list(lv.values()) if isinstance(lv, dict) else list(lv)


# ----------------------------------------------------------------------
# Un run = une methode, une seed
# ----------------------------------------------------------------------
def run_one(method, seed, eval_specs, sources, n_ood_list, alphas, cfg, use_ts):
    K = cfg["dataset"]["n_classes"]
    ccfg = cfg["conformal"]
    min_samples = ccfg["class_conditional_min_samples"]
    split_seed = 0 if seed is None else seed

    cal_id = load_method(method, "calib", seed, use_ts)
    cal_ood = load_method(method, "ood_calib_pool", seed, use_ts) if any(s != "id" for s in sources) else None

    data = {(sp, sg): load_method(method, sp, seed, use_ts, sg) for sp, sg in eval_specs}

    prob_rows, rows = [], []
    for (sp, sg), (p, y) in data.items():
        prob_rows.append({"method": method, "seed": seed, "split": sp, "sigma": sg,
                          **evaluate_probs(p, y), "n": int(len(y))})

    for source in sources:
        sizes = [None] if source == "id" else n_ood_list
        for n_ood in sizes:
            if source != "id" and n_ood == 0:
                continue
            cp, cy = build_calibration_set(source, cal_id, cal_ood, n_ood, split_seed)
            for score, cond, alpha in itertools.product(ccfg["scores"], ccfg["conditioning"], alphas):
                cal = calibrate(cp, cy, alpha, score, cond, min_samples)
                for (sp, sg), (p, y) in data.items():
                    m = evaluate_sets(predict_sets(p, cal), y, K)
                    rows.append({"method": method, "seed": seed, "split": sp, "sigma": sg,
                                 "score": score, "conditioning": cond, "source": source,
                                 "n_ood": n_ood, "alpha": alpha, "n_cal": cal.n_cal,
                                 "fallback_classes": int(cal.fallback.sum()), **m})
    return rows, prob_rows


def _mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def _std(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(v)) if v else float("nan")


def _groups(rows, alpha):
    """Regroupe les lignes (une par seed) par configuration, pour un alpha donne."""
    groups = {}
    for r in rows:
        if r["alpha"] != alpha:
            continue
        k = (r["method"], r["split"], r["sigma"] or 0, r["source"], r["n_ood"] or 0,
             r["score"], r["conditioning"])
        groups.setdefault(k, []).append(r)
    return dict(sorted(groups.items(), key=lambda kv: str(kv[0])))


def _label(k):
    name = k[1] + (_sig(k[2]) if k[2] else "")
    src = k[3] + (f"({k[4]})" if k[4] else "")
    return name, src


def summarize_probs(prob_rows):
    """Metriques du modele seul (sans ensembles) : accuracy, F1 macro, NLL, ECE, par split."""
    print("\nModele (probabilites, temperature scaling applique) : moyenne +- ecart-type sur les seeds")
    print(f"{'methode':10s} {'split':20s} {'n':>5s} {'accuracy':>16s} {'F1 macro':>16s} {'NLL':>14s} {'ECE':>14s}")
    groups = {}
    for r in prob_rows:
        groups.setdefault((r["method"], r["split"], r["sigma"] or 0), []).append(r)
    for k, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        name = k[1] + (_sig(k[2]) if k[2] else "")
        f = lambda key: f"{_mean([x[key] for x in g]):6.3f} +- {_std([x[key] for x in g]):5.3f}"
        print(f"{k[0]:10s} {name:20s} {g[0]['n']:5d} {f('acc'):>16s} {f('macro_f1'):>16s} "
              f"{f('nll'):>14s} {f('ece'):>14s}")


def summarize(rows, alpha_main):
    """Tableau principal : couverture (moyenne +- ecart-type), taille, singletons, vides, abstention, risque selectif."""
    print(f"\nEnsembles de prediction (alpha = {alpha_main} ; cible de couverture {1 - alpha_main:.2f} ; "
          f"moyenne sur les seeds)")
    print(f"{'methode':9s} {'split':19s} {'source':13s} {'score':5s} {'conditionnement':17s} "
          f"{'couverture':>15s} {'taille':>6s} {'singl':>6s} {'vide':>6s} {'abst':>6s} {'risque_sel':>10s}")
    for k, g in _groups(rows, alpha_main).items():
        name, src = _label(k)
        cov = [x["coverage"] for x in g]
        print(f"{k[0]:9s} {name:19s} {src:13s} {k[5]:5s} {k[6]:17s} "
              f"{_mean(cov):8.3f} +- {_std(cov):4.3f} "
              f"{_mean([x['mean_set_size'] for x in g]):6.2f} {_mean([x['singleton_rate'] for x in g]):6.3f} "
              f"{_mean([x['empty_rate'] for x in g]):6.3f} {_mean([x['abstention_rate'] for x in g]):6.3f} "
              f"{_mean([x['selective_risk'] for x in g]):10.3f}")


def summarize_per_class(rows, alpha_main, K=8, min_n=30):
    """Couverture par classe (moyenne sur les seeds). '*' = moins de min_n images : indicatif seulement."""
    print(f"\nCouverture par classe (alpha = {alpha_main} ; cible {1 - alpha_main:.2f} ; "
          f"* = moins de {min_n} images dans la classe : indicatif)")
    print(f"{'split':19s} {'source':13s} {'score':5s} {'conditionnement':17s} "
          + " ".join(f"{'c' + str(c):>6s}" for c in range(K)))
    shown = set()
    for k, g in _groups(rows, alpha_main).items():
        name, src = _label(k)
        n_cls = g[0]["n_per_class"]
        cells = []
        for c in range(K):
            m = _mean([x["coverage_per_class"][c] for x in g])
            cells.append("   n/a" if np.isnan(m) else f"{m:5.2f}" + ("*" if n_cls[c] < min_n else " "))
        print(f"{name:19s} {src:13s} {k[5]:5s} {k[6]:17s} " + " ".join(cells))
        if (name, src) not in shown:
            shown.add((name, src))
    # effectifs de chaque split (une seule fois par split)
    print("\nEffectifs par classe :")
    seen = set()
    for k, g in _groups(rows, alpha_main).items():
        name, _ = _label(k)
        if name in seen:
            continue
        seen.add(name)
        print(f"  {name:19s} " + " ".join(f"{n:6d}" for n in g[0]["n_per_class"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--mode", choices=["smoke", "final"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["single"],
                    choices=["single", "mcdropout", "ensemble"])
    ap.add_argument("--no_ts", action="store_true", help="ablation : sans temperature scaling")
    ap.add_argument("--n_ood", type=int, nargs="+", default=None,
                    help="tailles du pool OOD (ablation) ; defaut : pool complet")
    ap.add_argument("--tag", default="", help="suffixe du fichier de sortie")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["conformal"]
    seeds = args.seeds or cfg["seeds"]["training"]
    use_ts = not args.no_ts
    commit, dirty = git_state()

    tag = args.tag
    if args.no_ts:
        tag += "_no_ts"
    if args.n_ood:
        tag += "_nood" + "-".join(map(str, args.n_ood))
    if args.methods != ["single"]:
        tag += "_" + "-".join(args.methods)
    res_dir = ensure_dir("outputs/results")
    out_path = res_dir / f"{args.mode}_conformal{tag}.json"

    if args.mode == "smoke":
        eval_specs = [("val", None)]
        sources = ["id"]
        alphas = [ccfg["alpha_main"]]
        if args.n_ood or args.methods != ["single"]:
            print("Note : en smoke, seule la methode 'single' et la source 'id' sont testees.")
        args.methods = ["single"]
    else:
        sig = sigma_values(cfg)
        if not sig:
            sys.exit("sigma_levels est null dans protocol.yaml : fige le protocole v1.1 d'abord.")
        if out_path.exists():
            sys.exit(f"{out_path} existe deja : le mode final ne s'execute qu'une fois par configuration.")
        if commit is None:
            sys.exit("git indisponible : impossible de prouver quelle version du code produit les resultats.")
        if dirty:
            sys.exit("L'arbre git a des modifications non commitees : commite (git status doit etre propre).")
        eval_specs = [("test_id", None)] + [("test_id", s) for s in sig] + [("test_ood", None)]
        sources = list(ccfg["calibration_sources"].keys())
        alphas = [ccfg["alpha_main"]] + list(ccfg["alpha_sensitivity"])

    n_ood_list = args.n_ood or [None]
    all_rows, all_prob = [], []
    for method in args.methods:
        run_seeds = [None] if method == "ensemble" else seeds
        for seed in run_seeds:
            print(f"[{args.mode}] methode={method} seed={seed} ...")
            rows, prob_rows = run_one(method, seed, eval_specs, sources, n_ood_list, alphas, cfg, use_ts)
            all_rows += rows
            all_prob += prob_rows

    out = {"mode": args.mode, "git_commit": commit, "git_dirty": dirty, "temperature_scaling": use_ts,
           "methods": args.methods, "seeds": seeds, "alphas": alphas, "n_ood": args.n_ood,
           "protocol_revision": cfg["dataset"].get("revision"),
           "probs_metrics": all_prob, "conformal": all_rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summarize_probs(all_prob)
    for a in alphas:
        summarize(all_rows, a)
    summarize_per_class(all_rows, ccfg["alpha_main"], cfg["dataset"]["n_classes"])
    print(f"\nResultats : {out_path}")


if __name__ == "__main__":
    main()