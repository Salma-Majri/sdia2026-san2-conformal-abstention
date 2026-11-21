"""Prédiction conforme fractionnée (LAC et APS), marginale et par classe.

Fournit les fonctions de base pour run_conformal.py
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
LOGITS_DIR = ROOT / "outputs" / "logits"

SCORES = ("lac", "aps")
CONDITIONINGS = ("marginal", "class_conditional")
SOURCES = ("id", "ood_few_shot", "mixed")


def load_temperature(seed):
    path = ROOT / "outputs" / "results" / f"metrics_val_seed{seed}.json"
    with open(path, "r") as f:
        return json.load(f)["temperature"]


def load_probs(split, seed, T=1.0, sigma=None):
    suffix = f"_sigma{sigma:g}" if sigma else ""
    path = LOGITS_DIR / f"{split}_seed{seed}{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    logits = np.load(path).astype(np.float64)
    labels = np.load(LOGITS_DIR / f"{split}_labels.npy").astype(np.int64)
    return softmax(logits / T, axis=1), labels


def score_matrix(probs, score="lac"):
    probs = np.asarray(probs, dtype=np.float64)
    if score == "lac":
        return 1.0 - probs
    if score == "aps":
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        out = np.empty_like(cum)
        np.put_along_axis(out, order, cum, axis=1)
        return out
    raise ValueError(f"score inconnu : {score}")


def true_class_scores(probs, y, score="lac"):
    return score_matrix(probs, score)[np.arange(len(y)), y]


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(s[k - 1])


@dataclass
class Calibrated:
    score: str
    conditioning: str
    alpha: float
    qhat: np.ndarray
    q_global: float
    n_cal: int
    n_per_class: np.ndarray
    fallback: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def calibrate(probs, y, alpha, score="lac", conditioning="marginal", min_samples=20):
    probs = np.asarray(probs)
    y = np.asarray(y).astype(np.int64)
    K = probs.shape[1]
    s = true_class_scores(probs, y, score)
    q_global = conformal_quantile(s, alpha)
    counts = np.bincount(y, minlength=K)

    qhat = np.full(K, q_global)
    fallback = np.ones(K, dtype=bool)
    if conditioning == "class_conditional":
        for c in range(K):
            if counts[c] >= min_samples:
                qhat[c] = conformal_quantile(s[y == c], alpha)
                fallback[c] = False
    else:
        fallback[:] = False
    return Calibrated(score, conditioning, float(alpha), qhat, q_global, len(y), counts, fallback)


def predict_sets(probs, cal):
    S = score_matrix(probs, cal.score)
    return S <= cal.qhat[None, :]


def build_calibration_set(source, id_probs, ood_probs, n_ood=None, seed=0):
    if source == "id":
        return np.asarray(id_probs[0]), np.asarray(id_probs[1])
    if ood_probs is None:
        raise ValueError("ood_probs requis pour ood_few_shot et mixed")
    n = len(ood_probs[1]) if n_ood is None else int(n_ood)
    idx = np.random.default_rng(seed).choice(len(ood_probs[1]), size=n, replace=False)
    if source == "ood_few_shot":
        return ood_probs[0][idx], ood_probs[1][idx]
    return (np.concatenate([id_probs[0], ood_probs[0][idx]]),
            np.concatenate([id_probs[1], ood_probs[1][idx]]))