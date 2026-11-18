"""
make_figures.py - Figures de l'etat actuel du projet (aucun jeu de test evalue).

Une figure = un fichier (PNG + PDF dans outputs/figures/). La legende est placee
sous le graphe, jamais sur les courbes.

  fig_loss              : loss train (tirets) et val (plein), 3 seeds
  fig_val_accuracy      : accuracy de validation par epoque, 3 seeds
  fig_val_f1            : F1 macro de validation par epoque, 3 seeds
  fig_class_pct         : repartition des classes par split (%)
  fig_class_counts      : effectifs par classe (echelle log) et seuil de repli
  fig_noise_pilot       : chute d'accuracy sur val en fonction de sigma

Usage (depuis la racine du projet) :
    python scripts/make_figures.py

N'utilise que : historiques d'entrainement, labels des splits, resultats du pilote (val).
Ne lit AUCUN logit de test_id, test_id bruite ni test_ood.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import load_config  # noqa: E402

RES = ROOT / "outputs" / "results"
LOG = ROOT / "outputs" / "logits"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
FIGSIZE = (8, 4.6)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"OK  {OUT / name}.png")


def legend_below(ax, handles=None, ncol=3):
    """Legende sous le graphe (ne cache ni courbes ni barres)."""
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=ncol,
              frameon=False, fontsize=10)
    if handles is not None:
        ax.legend(handles=handles, **kw)
    else:
        ax.legend(**kw)


# ----------------------------------------------------------------------
# 1-3. Courbes d'entrainement (une figure par metrique)
# ----------------------------------------------------------------------
def _load_runs():
    runs = {}
    for s in SEEDS:
        p = RES / f"train_main_seed{s}.json"
        if p.exists():
            runs[s] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def _curve_figure(runs, name, title, ylabel, val_key, train_key=None):
    max_ep = max(len(r["history"]) for r in runs.values())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s, r in runs.items():
        h = r["history"]
        ep = [x["epoch"] for x in h]
        val = [x[val_key] for x in h]
        c = COLORS[s]
        ax.plot(ep, val, "-", color=c, lw=2)
        if train_key:
            ax.plot(ep, [x[train_key] for x in h], "--", color=c, lw=1.5, alpha=0.7)
        best = int(np.argmin([x["val_loss"] for x in h]))   # epoque retenue (val_loss min)
        ax.plot(ep[best], val[best], "o", color=c, ms=9, mfc="white", mew=2.2, zorder=5)

    ax.set_xlim(0.5, max_ep + 0.5)
    ax.set_xticks(range(1, max_ep + 1))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Époque")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)

    handles = [Line2D([0], [0], color=COLORS[s], lw=2, label=f"seed {s}") for s in runs]
    if train_key:
        handles += [Line2D([0], [0], color="k", lw=2, ls="-", label="validation"),
                    Line2D([0], [0], color="k", lw=1.5, ls="--", label="entraînement")]
    handles.append(Line2D([0], [0], marker="o", color="k", mfc="white", mew=2, ls="",
                          ms=8, label="époque retenue (val_loss min)"))
    legend_below(ax, handles, ncol=3)
    save(fig, name)


def fig_training_curves():
    runs = _load_runs()
    if not runs:
        print("SKIP courbes : aucun train_main_seed*.json dans outputs/results/")
        return
    _curve_figure(runs, "fig_loss", "Loss : entraînement et validation",
                  "Loss (cross-entropy)", "val_loss", "train_loss")
    _curve_figure(runs, "fig_val_accuracy", "Accuracy sur la validation",
                  "Accuracy", "val_acc")
    _curve_figure(runs, "fig_val_f1", "F1 macro sur la validation",
                  "F1 macro", "val_macro_f1")

    print("\nResume entrainement (meilleure epoque sur val_loss) :")
    for s, r in runs.items():
        h = r["history"]
        b = min(h, key=lambda x: x["val_loss"])
        print(f"  seed {s} : epoque {b['epoch']}/{len(h)}  val_loss {b['val_loss']:.4f}  "
              f"acc {b['val_acc']:.4f}  F1 {b['val_macro_f1']:.4f}")


# ----------------------------------------------------------------------
# 4-5. Distribution des classes par split (labels seulement)
# ----------------------------------------------------------------------
def fig_class_distribution():
    splits = ["val", "calib", "test_id", "ood_calib_pool", "test_ood"]
    counts = {}
    for sp in splits:
        p = LOG / f"{sp}_labels.npy"
        if p.exists():
            counts[sp] = np.bincount(np.load(p), minlength=8)
    if not counts:
        print("SKIP distribution : aucun *_labels.npy dans outputs/logits/")
        return

    names = list(counts)
    x = np.arange(8)
    w = 0.8 / len(names)

    def grouped(ax, values_fn):
        for i, sp in enumerate(names):
            c = counts[sp]
            ax.bar(x + i * w, values_fn(c), w, label=f"{sp} (n={c.sum()})")
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([f"classe {k}" for k in range(8)], fontsize=9)

    # Pourcentages
    fig, ax = plt.subplots(figsize=FIGSIZE)
    grouped(ax, lambda c: 100 * c / c.sum())
    ax.set_ylabel("Part du split (%)")
    ax.set_title("Répartition des classes par split", pad=12)
    legend_below(ax, ncol=3)
    save(fig, "fig_class_pct")

    # Effectifs en echelle log
    fig, ax = plt.subplots(figsize=FIGSIZE)
    grouped(ax, lambda c: c)
    ax.set_yscale("log")
    ax.axhline(20, color="red", ls="--", lw=1.5,
               label="seuil 20 (repli du conforme par classe)")
    ax.set_ylabel("Effectif (échelle log)")
    ax.set_title("Effectifs par classe et par split", pad=12)
    legend_below(ax, ncol=2)
    save(fig, "fig_class_counts")


# ----------------------------------------------------------------------
# 6. Pilote du bruit (val uniquement)
# ----------------------------------------------------------------------
def _find_rows(obj):
    """Cherche une liste de (sigma, accuracy) dans un JSON de structure inconnue."""
    sig_keys = ("sigma", "noise_sigma")
    acc_keys = ("acc", "accuracy", "val_acc")
    if isinstance(obj, list) and obj and all(isinstance(o, dict) for o in obj):
        sk = next((k for k in sig_keys if all(k in o for o in obj)), None)
        ak = next((k for k in acc_keys if all(k in o for o in obj)), None)
        if sk and ak:
            return sorted((float(o[sk]), float(o[ak])) for o in obj)
    if isinstance(obj, dict):
        try:
            rows = []
            for k, v in obj.items():
                val = v if isinstance(v, (int, float)) else next(
                    (v[a] for a in acc_keys if isinstance(v, dict) and a in v), None)
                if val is None:
                    raise ValueError
                rows.append((float(k), float(val)))
            if rows:
                return sorted(rows)
        except (ValueError, TypeError):
            pass
        for v in obj.values():
            r = _find_rows(v)
            if r:
                return r
    return None


def fig_noise_pilot(cfg):
    p = RES / "pilot_noise_seed0.json"
    if not p.exists():
        print("SKIP pilote : outputs/results/pilot_noise_seed0.json absent")
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = _find_rows(data)
    if not rows:
        top = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        print(f"SKIP pilote : format du JSON non reconnu (cles : {top}). "
              "Adapte _find_rows() a ton format.")
        return

    sig = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    if 0.0 in sig:
        base = acc[sig.index(0.0)]
    else:   # accuracy propre sur val, seed 0 (val uniquement)
        lp, lb = LOG / "val_seed0.npy", LOG / "val_labels.npy"
        if not (lp.exists() and lb.exists()):
            print("SKIP pilote : pas d'accuracy propre (sigma=0) disponible.")
            return
        base = float((np.load(lp).argmax(1) == np.load(lb)).mean())
    pts = [(s, a) for s, a in zip(sig, acc) if s > 0]
    xs = [s for s, _ in pts]
    drops = [100 * (base - a) for _, a in pts]

    an = cfg["shift_simulation"]["acquisition_noise"]
    levels = an.get("sigma_levels")
    targets = an.get("target_accuracy_drop_points", {})
    level_colors = {"leger": "#2ca02c", "moyen": "#ff9900", "fort": "#d62728"}
    pretty = {"leger": "léger", "moyen": "moyen", "fort": "fort"}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, drops, "o-", color="#1f77b4", lw=2, ms=6,
            label="chute d'accuracy mesurée (val, seed 0)")
    if levels:
        items = levels.items() if isinstance(levels, dict) else enumerate(levels)
        for k, s in items:
            c = level_colors.get(k, "red")
            t = targets.get(k)
            ax.axvline(s, color=c, ls="--", lw=1.5)
            if t is not None:
                ax.axhline(t, color=c, ls=":", lw=1.2, alpha=0.8)
            lab = f"{pretty.get(k, k)} : σ = {s:g}" + (f" (cible {t} pts)" if t is not None else "")
            ax.plot([], [], color=c, ls="--", lw=1.5, label=lab)

    ax.set_xlabel("σ (écart-type du bruit gaussien, image en [0, 1])")
    ax.set_ylabel("Chute d'accuracy (points)")
    ax.set_title(f"Pilote du bruit sur val (accuracy propre : {100 * base:.1f} %)", pad=12)
    legend_below(ax, ncol=2)
    save(fig, "fig_noise_pilot")


def main():
    cfg = load_config()
    fig_training_curves()
    fig_class_distribution()
    fig_noise_pilot(cfg)


if __name__ == "__main__":
    main()