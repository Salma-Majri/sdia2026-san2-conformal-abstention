"""
make_uncertainty_figures.py - Génère toutes les figures d'incertitude pour l'article.
Usage : python scripts/make_uncertainty_figures.py
Ce script utilise le module src/uncertainty_metrics.py pour calculer les métriques
et génère des graphiques publication-ready dans outputs/figures/uncertainty/.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ajoute src au path pour importer les modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils import ROOT
from uncertainty_metrics import shannon_entropy, expected_calibration_error, risk_coverage_curve

# Configuration des chemins
OUTPUT_DIR = ROOT / "outputs"
LOGITS_DIR = OUTPUT_DIR / "logits"
MCD_DIR = OUTPUT_DIR / "mcdropout"
ENS_DIR = OUTPUT_DIR / "ensembles"
FIG_DIR = OUTPUT_DIR / "figures" / "uncertainty"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style professionnel
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300


def load_val_data():
    """Charge les prédictions sur val."""
    data = {}
    # Single
    single_probs = []
    for seed in [0, 1, 2]:
        path = LOGITS_DIR / f"val_seed{seed}.npy"
        if path.exists():
            logits = np.load(path).astype(np.float64)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            single_probs.append(probs)
    if single_probs:
        data['single'] = np.mean(single_probs, axis=0)
    
    # MC-Dropout
    mcd_probs = []
    for seed in [0, 1, 2]:
        path = MCD_DIR / f"mcdropout_val_seed{seed}.npy"
        if path.exists():
            mcd_probs.append(np.load(path).astype(np.float64))
    if mcd_probs:
        data['mc_dropout'] = np.mean(mcd_probs, axis=0)
        
    # Ensemble
    ens_path = ENS_DIR / "ensemble_val.npy"
    if ens_path.exists():
        data['ensemble'] = np.load(ens_path).astype(np.float64)
        
    labels_path = LOGITS_DIR / "val_labels.npy"
    if labels_path.exists():
        data['labels'] = np.load(labels_path).astype(np.int64)
        
    return data


def plot_entropy_distribution(data):
    """FIGURE 1 : Distribution de l'entropie."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'single': '#2196F3', 'mc_dropout': '#FF9800', 'ensemble': '#4CAF50'}
    
    for method, probs in data.items():
        if method == 'labels': continue
        entropy = shannon_entropy(probs)
        sns.kdeplot(entropy, ax=ax, label=method.replace('_', ' ').title(), 
                    fill=True, alpha=0.4, color=colors.get(method, 'gray'), linewidth=2)

    ax.set_xlabel("Entropie de la prédiction (nats)", fontsize=13)
    ax.set_ylabel("Densité", fontsize=13)
    ax.set_title("Distribution de l'incertitude sur val", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_entropy_distribution_val.png", bbox_inches='tight')
    print(" Figure 1 sauvegardée : fig1_entropy_distribution_val.png")


def plot_reliability_diagrams(data):
    """FIGURE 2 : Diagrammes de fiabilité (ECE)."""
    methods = [k for k in data.keys() if k != 'labels']
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5), sharey=True)
    if len(axes) == 1: axes = [axes]

    for ax, method in zip(axes, methods):
        ece, bins = expected_calibration_error(data[method], data['labels'])
        confidences = [b["confidence"] for b in bins]
        accuracies = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]
        
        ax.bar(confidences, counts, width=1/15, alpha=0.3, color='gray', edgecolor='black')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Parfaite')
        ax.plot(confidences, accuracies, 'o-', linewidth=2, color='#E91E63', label=f'ECE={ece:.4f}')
        
        ax.set_title(method.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Accuracy / Effectif", fontsize=12)
    fig.suptitle("Diagrammes de fiabilité sur val (15 bins)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_reliability_diagram_val.png", bbox_inches='tight')
    print("✅ Figure 2 sauvegardée : fig2_reliability_diagram_val.png")


def plot_risk_coverage_curves(data):
    """FIGURE 3 : Courbes Risque-Couverture."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'single': '#2196F3', 'mc_dropout': '#FF9800', 'ensemble': '#4CAF50'}
    
    for method, probs in data.items():
        if method == 'labels': continue
        coverages, risks = risk_coverage_curve(probs, data['labels'])
        ax.plot(coverages, risks, 'o-', linewidth=2, markersize=4, 
                label=method.replace('_', ' ').title(), color=colors.get(method, 'gray'))

    ax.set_xlabel("Couverture (taux de cas gardés)", fontsize=13)
    ax.set_ylabel("Risque (taux d'erreur)", fontsize=13)
    ax.set_title("Courbes Risque-Couverture sur val", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_risk_coverage_curves_val.png", bbox_inches='tight')
    print(" Figure 3 sauvegardée : fig3_risk_coverage_curves_val.png")


def main():
    print(" Génération des figures d'incertitude...")
    data = load_val_data()
    if not data:
        print(" Aucune donnée trouvée. Lance d'abord mc_dropout.py et ensembles.py sur val.")
        return
    
    print(f" Méthodes chargées : {[k for k in data.keys() if k != 'labels']}")
    plot_entropy_distribution(data)
    plot_reliability_diagrams(data)
    plot_risk_coverage_curves(data)
    print(" Toutes les figures sont prêtes dans outputs/figures/uncertainty/")


if __name__ == "__main__":
    main()