"""
make_conformal_figures.py - Figures a partir des resultats du run final (aucun calcul sur le test :
on ne fait que lire les JSON de outputs/results/).

Lit :   outputs/results/final_conformal.json                     (obligatoire)
        outputs/results/final_conformal_no_ts.json               (optionnel : ablation temperature)
        outputs/results/final_conformal_nood50-100-361.json      (optionnel : ablation taille du pool OOD)
Ecrit : outputs/figures/conformal_*.png  (un graphique par fichier, legende sous le graphique)

    python scripts/make_conformal_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
ALPHA = 0.10
TARGET = 1 - ALPHA

CONFIGS = [("lac", "marginal"), ("lac", "class_conditional"),
           ("aps", "marginal"), ("aps", "class_conditional")]
LABEL = {("lac", "marginal"): "LAC marginal", ("lac", "class_conditional"): "LAC par classe",
         ("aps", "marginal"): "APS marginal", ("aps", "class_conditional"): "APS par classe"}
COLOR = {("lac", "marginal"): "#9ecae1", ("lac", "class_conditional"): "#08519c",
         ("aps", "marginal"): "#fdd0a2", ("aps", "class_conditional"): "#d94801"}
SRC_LABEL = {"id": "calibration ID", "mixed": "calibration mixte", "ood_few_shot": "calibration OOD (few-shot)"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


# ----------------------------------------------------------------------
def load(name):
    p = RES / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def mean_std(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0) if len(v) else (np.nan, 0.0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("ecrit :", FIG / name)


def legend_below(ax, ncol=4):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol, frameon=False)


def grouped_bars(ax, cats, series, ylabel, ylim=None, hline=None):
    """series : {config: (means, stds)} ; une barre par config dans chaque categorie."""
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(cats))
    for i, (cfg, (m, s)) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, m, w, yerr=s, capsize=2, color=COLOR[cfg],
               edgecolor="black", linewidth=0.4, label=LABEL[cfg], error_kw={"elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1)


# ----------------------------------------------------------------------
def scenarios(d, source="id"):
    """[(nom, split, sigma)] dans l'ordre : ID, bruit leger/moyen/fort, nouveau centre."""
    sig = sorted({r["sigma"] for r in d["conformal"] if r["split"] == "test_id" and r["sigma"]})
    out = [("ID", "test_id", None)]
    out += [(f"bruit σ={s:g}", "test_id", s) for s in sig]
    out += [("nouveau centre", "test_ood", None)]
    return out


def fig_coverage_shifts(d):
    rows = sel(d["conformal"], alpha=ALPHA, source="id")
    sc = scenarios(d)
    series = {}
    for cfg in CONFIGS:
        ms = [mean_std(sel(rows, split=sp, sigma=sg, score=cfg[0], conditioning=cfg[1]), "coverage") for _, sp, sg in sc]
        series[cfg] = ([m for m, _ in ms], [s for _, s in ms])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    grouped_bars(ax, [n for n, _, _ in sc], series, "couverture empirique", (0.5, 1.0), TARGET)
    ax.set_title("Couverture sous décalage (calibration ID, cible 0.90)")
    legend_below(ax)
    save(fig, "conformal_coverage_vs_shift.png")


def fig_abstention_shifts(d):
    rows = sel(d["conformal"], alpha=ALPHA, source="id")
    sc = scenarios(d)
    series = {}
    for cfg in CONFIGS:
        ms = [mean_std(sel(rows, split=sp, sigma=sg, score=cfg[0], conditioning=cfg[1]), "abstention_rate") for _, sp, sg in sc]
        series[cfg] = ([m for m, _ in ms], [s for _, s in ms])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    grouped_bars(ax, [n for n, _, _ in sc], series, "taux d'abstention (taille ≠ 1)", (0, 1.0))
    ax.set_title("Abstention sous décalage (calibration ID)")
    legend_below(ax)
    save(fig, "conformal_abstention_shifts.png")


def fig_noise_selective_risk(d):
    rows = sel(d["conformal"], alpha=ALPHA, source="id", split="test_id")
    sig = [None] + sorted({r["sigma"] for r in rows if r["sigma"]})
    xs = np.arange(len(sig))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cfg in CONFIGS:
        ms = [mean_std(sel(rows, sigma=s, score=cfg[0], conditioning=cfg[1]), "selective_risk") for s in sig]
        ax.errorbar(xs, [m for m, _ in ms], [s for _, s in ms], marker="o", capsize=3,
                    color=COLOR[cfg], label=LABEL[cfg], linestyle="-" if cfg[1] == "class_conditional" else "--")
    ax.set_xticks(xs)
    ax.set_xticklabels(["sans bruit"] + [f"σ={s:g}" for s in sig[1:]])
    ax.set_ylabel("risque sélectif (erreur parmi les singletons)")
    ax.set_title("Le bruit dégrade les prédictions retenues, pas l'abstention")
    legend_below(ax, 2)
    save(fig, "conformal_noise_selective_risk.png")


def fig_ood_sources(d, metric="coverage", name="conformal_ood_sources_coverage.png", ylabel="couverture empirique sur test_ood",
                    ylim=(0.5, 1.0), hline=TARGET):
    rows = sel(d["conformal"], alpha=ALPHA, split="test_ood", sigma=None)
    rows = [r for r in rows if r["n_ood"] in (None,)]
    srcs = ["id", "mixed", "ood_few_shot"]
    series = {}
    for cfg in CONFIGS:
        ms = [mean_std(sel(rows, source=s, score=cfg[0], conditioning=cfg[1]), metric) for s in srcs]
        series[cfg] = ([m for m, _ in ms], [s for _, s in ms])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bars(ax, [SRC_LABEL[s] for s in srcs], series, ylabel, ylim, hline)
    ax.set_title("Nouveau centre : couverture selon la source de calibration" if metric == "coverage"
                 else "Nouveau centre : coût en abstention selon la source de calibration")
    legend_below(ax)
    save(fig, name)


def fig_nood(dn):
    rows = sel(dn["conformal"], alpha=ALPHA, split="test_ood", sigma=None)
    sizes = [0, 50, 100, 361]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for cfg in CONFIGS:
        ys, es = [], []
        for n in sizes:
            r = (sel(rows, source="id", score=cfg[0], conditioning=cfg[1]) if n == 0
                 else sel(rows, source="ood_few_shot", n_ood=n, score=cfg[0], conditioning=cfg[1]))
            m, s = mean_std(r, "coverage")
            ys.append(m)
            es.append(s)
        ax.errorbar([0], ys[:1], es[:1], marker="s", capsize=3, color=COLOR[cfg], linestyle="none")
        ax.errorbar(range(1, len(sizes)), ys[1:], es[1:], marker="o", capsize=3, color=COLOR[cfg],
                    label=LABEL[cfg], linestyle="-" if cfg[1] == "class_conditional" else "--")
    ax.axhline(TARGET, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("images OOD dans la calibration (0 = calibration ID seule, carré ; ± écart-type sur 3 seeds)")
    ax.set_ylabel("couverture sur test_ood")
    ax.set_title("Taille du pool OOD (calibration OOD few-shot)")
    legend_below(ax)
    save(fig, "conformal_ablation_nood.png")


def fig_per_class(d, split="test_id"):
    rows = sel(d["conformal"], alpha=ALPHA, split=split, sigma=None, source="id")
    K = len(rows[0]["coverage_per_class"])
    n_pc = np.mean([r["n_per_class"] for r in sel(rows, score="lac", conditioning="marginal")], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.8 / len(CONFIGS)
    for i, cfg in enumerate(CONFIGS):
        r = sel(rows, score=cfg[0], conditioning=cfg[1])
        a = np.array([x["coverage_per_class"] for x in r], dtype=float)
        mu = a.mean(0)
        sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(K)
        ax.bar(np.arange(K) + (i - 1.5) * w, mu, w, yerr=[np.minimum(sd, mu), np.minimum(sd, 1.0 - mu)],
               capsize=2, color=COLOR[cfg], edgecolor="black", linewidth=0.4, label=LABEL[cfg],
               error_kw={"elinewidth": 0.8})
    ax.axhline(TARGET, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"c{k}\n(n={int(n_pc[k])}{'*' if n_pc[k] < 30 else ''})" for k in range(K)])
    ax.set_ylabel("couverture par classe (± écart-type sur 3 seeds)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Couverture par classe sur test_id (* = moins de 30 images : indicatif)")
    legend_below(ax)
    save(fig, f"conformal_coverage_per_class_{split}.png")


def fig_prevalence(d):
    rows = sel(d["prevalence_shift"], alpha=ALPHA)
    scen = ["uniform", "inverse", "rare_boost"]
    series = {}
    for cfg in CONFIGS:
        ms, ss = [], []
        for s in scen:
            r = sel(rows, scenario=s, score=cfg[0], conditioning=cfg[1])
            ms.append(np.mean([x["coverage_mean"] for x in r]))
            ss.append(np.mean([x["coverage_std"] for x in r]))
        series[cfg] = (ms, ss)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    grouped_bars(ax, ["uniforme", "inverse", "classes rares ×5"], series, "couverture", (0.5, 1.0), TARGET)
    ax.set_title("Déséquilibre de prévalence (calibration ID inchangée)")
    legend_below(ax)
    save(fig, "conformal_prevalence_shift.png")


def fig_ece_ts(d, dno):
    labels = ["test_id", "σ=0.01", "σ=0.0125", "σ=0.02", "test_ood"]

    def vals(dd):
        out = []
        for sp, sg in [("test_id", None)] + [("test_id", s) for s in sorted({r["sigma"] for r in dd["probs_metrics"] if r["sigma"]})] + [("test_ood", None)]:
            out.append(mean_std(sel(dd["probs_metrics"], split=sp, sigma=sg), "ece"))
        return out
    a, b = vals(d), vals(dno)
    labels = labels[:len(a)]
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, [m for m, _ in b], 0.4, yerr=[s for _, s in b], capsize=2, color="#bdbdbd",
           edgecolor="black", linewidth=0.4, label="sans temperature scaling")
    ax.bar(x + 0.2, [m for m, _ in a], 0.4, yerr=[s for _, s in a], capsize=2, color="#31a354",
           edgecolor="black", linewidth=0.4, label="avec temperature scaling")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ECE")
    ax.set_title("Effet du temperature scaling sur la calibration des probabilités")
    legend_below(ax, 2)
    save(fig, "conformal_ablation_ts.png")


def main():
    d = load("final_conformal.json")
    if d is None:
        raise SystemExit("outputs/results/final_conformal.json introuvable : lancer le run final d'abord.")
    fig_coverage_shifts(d)          # 1. couverture : ID, bruit, nouveau centre
    fig_abstention_shifts(d)        # 2. cout en abstention
    fig_noise_selective_risk(d)     # 3. bruit : l'abstention n'augmente pas, le risque oui
    fig_ood_sources(d)              # 4. nouveau centre : calibration ID / mixte / OOD few-shot (couverture)
    fig_ood_sources(d, "abstention_rate", "conformal_ood_sources_abstention.png",
                    "taux d'abstention sur test_ood", (0, 1.0), None)   # 5. ... et son cout en abstention
    fig_per_class(d, "test_id")     # 6. couverture par classe (classes rares)
    fig_prevalence(d)               # 7. 3 scenarios de prevalence
    dno = load("final_conformal_no_ts.json")
    if dno:
        fig_ece_ts(d, dno)          # 8. ablation temperature scaling
    dn = load("final_conformal_nood50-100-361.json")
    if dn:
        fig_nood(dn)                # 9. ablation taille du pool OOD


if __name__ == "__main__":
    main()