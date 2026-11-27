"""
uncertainty_metrics.py - Métriques dédiées à l'incertitude et à l'abstention (SAN-2).
Module réutilisable par les scripts et notebooks de Fatima-Zahra.
Contrairement à src/metrics.py (Salma) qui gère Accuracy/F1/NLL, 
ce module se concentre sur l'entropie, l'ECE, le risque sélectif et les courbes risque-couverture.
"""
import numpy as np
from typing import Tuple, List

def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Calcule l'entropie de Shannon (en nats) pour chaque échantillon.
    Une entropie élevée = forte incertitude.
    """
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, List[dict]]:
    """
    Calcule l'Expected Calibration Error (ECE) et retourne les infos par bin 
    pour le tracé du diagramme de fiabilité.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_infos = []
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_size = mask.sum()
        
        if bin_size == 0:
            bin_infos.append({"confidence": 0, "accuracy": 0, "count": 0})
            continue
            
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        
        ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
        bin_infos.append({"confidence": bin_conf, "accuracy": bin_acc, "count": bin_size})
        
    return float(ece), bin_infos


def selective_risk(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Calcule le risque sélectif (taux d'erreur) et la couverture (taux de cas gardés) 
    pour un seuil de confiance donné.
    Retourne (risque, couverture). Si aucun cas n'est gardé, retourne (NaN, 0).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    
    mask = confidences >= threshold
    coverage = mask.mean()
    
    if mask.sum() == 0:
        return float('nan'), 0.0
        
    risk = 1.0 - (predictions[mask] == labels[mask]).mean()
    return float(risk), float(coverage)


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les points pour la courbe Risque-Couverture.
    Fait varier le seuil de confiance de la probabilité max minimale à 1.0.
    """
    confidences = probs.max(axis=1)
    min_conf = confidences.min()
    
    thresholds = np.linspace(min_conf, 0.999, n_points)
    risks = []
    coverages = []
    
    for t in thresholds:
        risk, cov = selective_risk(probs, labels, t)
        risks.append(risk)
        coverages.append(cov)
        
    return np.array(coverages), np.array(risks)