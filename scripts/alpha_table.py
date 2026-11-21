"""
alpha_table.py - Tableau de sensibilite a alpha (0.20 / 0.10 / 0.05), a partir de
outputs/results/final_conformal.json. Aucun calcul sur le test : on ne fait que lire le JSON.

Pour chaque jeu d'evaluation (test_id, bruit, test_ood) et chaque methode (LAC/APS x marginal/par classe) :
couverture (moyenne +- ecart-type sur les seeds) et abstention, pour la source de calibration choisie.

    python scripts/alpha_table.py                         # calibration ID
    python scripts/alpha_table.py --source ood_few_shot
Ecrit : outputs/results/alpha_table_<source>.csv  et  .md
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
ALPHAS = [0.2, 0.1, 0.05]
CONFIGS = [("lac", "marginal", "LAC marginal"), ("lac", "class_conditional", "LAC par classe"),
           ("aps", "marginal", "APS marginal"), ("aps", "class_conditional", "APS par classe")]


def mean_std(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="id", choices=["id", "mixed", "ood_few_shot"])
    args = ap.parse_args()

    d = json.load(open(RES / "final_conformal.json", encoding="utf-8"))
    rows = [r for r in d["conformal"] if r["source"] == args.source and r["n_ood"] is None]

    evals = sorted({(r["split"], r["sigma"]) for r in rows},
                   key=lambda k: (0 if (k[0] == "test_id" and not k[1]) else 1 if k[0] == "test_id" else 2, k[1] or 0))

    def name(k):
        return k[0] if not k[1] else f"{k[0]} + bruit sigma={k[1]:g}"

    header = ["evaluation", "methode"] + [f"couv a={a:g} (cible {1 - a:.2f})" for a in ALPHAS] + \
             [f"abst a={a:g}" for a in ALPHAS]
    out = []
    for k in evals:
        for score, cond, label in CONFIGS:
            cov, ab = [], []
            for a in ALPHAS:
                sel = [r for r in rows if (r["split"], r["sigma"]) == k and r["score"] == score
                       and r["conditioning"] == cond and abs(r["alpha"] - a) < 1e-9]
                if not sel:
                    cov.append("-")
                    ab.append("-")
                    continue
                m, s = mean_std(sel, "coverage")
                cov.append(f"{m:.3f} +- {s:.3f}")
                ab.append(f"{mean_std(sel, 'abstention_rate')[0]:.3f}")
            out.append([name(k), label] + cov + ab)

    with open(RES / f"alpha_table_{args.source}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in out]
    (RES / f"alpha_table_{args.source}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nEcrit : outputs/results/alpha_table_{args.source}.csv et .md")


if __name__ == "__main__":
    main()