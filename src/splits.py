"""
splits.py - Protocole de splits OOD pour FedISIC2019 (flwrlabs/fed-isic2019).

Centres sources : train 0.65 / val 0.10 / calib 0.15 / test_id 0.10
Centre OOD      : ood_calib_pool 0.20 / test_ood 0.80
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# Stratification robuste
# ----------------------------------------------------------------------
def _min_count(strata):
    return min(Counter(strata.tolist()).values())


def _strata_levels(idx, centers, labels):
    """Genere des strates de plus en plus grossieres (None = pas de stratification)."""
    # Niveau 1 : centre_label, strates rares (<2) fusionnees par label
    s = np.array([f"{centers[i]}_{labels[i]}" for i in idx])
    cnt = Counter(s.tolist())
    merged = np.array([x if cnt[x] >= 2 else f"rare_{x.split('_')[1]}" for x in s])
    yield "centre_label (strates rares fusionnees)", merged
    # Niveau 2 : label seul
    yield "label", np.asarray(labels)[idx]
    # Niveau 3 : aucune stratification
    yield "aucune", None


def stratified_split(idx, test_size, seed, centers, labels, name=""):
    """train_test_split avec repli automatique si la stratification est impossible."""
    for level, strata in _strata_levels(idx, centers, labels):
        if strata is not None and _min_count(strata) < 2:
            continue
        try:
            a, b = train_test_split(
                idx, test_size=test_size, random_state=seed, stratify=strata
            )
            if level != "centre_label (strates rares fusionnees)":
                print(f"[{name}] stratification : {level}")
            return a, b
        except ValueError:
            continue
    raise RuntimeError(f"Impossible de decouper ({name})")


# ----------------------------------------------------------------------
# Protocole de splits
# ----------------------------------------------------------------------
def make_splits(centers, labels, main_ood_center=3, seed=0):
    centers = np.asarray(centers)
    labels = np.asarray(labels)
    indices = np.arange(len(labels), dtype=np.int64)

    source_idx = indices[centers != main_ood_center]
    ood_idx = indices[centers == main_ood_center]
    if len(ood_idx) == 0:
        raise ValueError(f"Aucune image pour le centre OOD {main_ood_center}")

    # Centres sources : 0.65 / 0.10 / 0.15 / 0.10
    train_idx, rest = stratified_split(source_idx, 0.35, seed, centers, labels, "train|rest")
    val_idx, tmp = stratified_split(rest, 0.25 / 0.35, seed, centers, labels, "val|tmp")
    calib_idx, test_id_idx = stratified_split(tmp, 0.10 / 0.25, seed, centers, labels, "calib|test_id")

    # Centre OOD : 20% calib pool / 80% test (sans fuite)
    ood_calib_idx, test_ood_idx = stratified_split(ood_idx, 0.80, seed, centers, labels, "ood_calib|test_ood")

    splits = {
        "train": train_idx,
        "val": val_idx,
        "calib": calib_idx,
        "test_id": test_id_idx,
        "ood_calib_pool": ood_calib_idx,
        "test_ood": test_ood_idx,
    }

    # Verifications strictes
    all_idx = np.concatenate(list(splits.values()))
    assert len(all_idx) == len(np.unique(all_idx)), "Chevauchement detecte !"
    assert len(all_idx) == len(indices), "Indices manquants !"
    assert set(centers[test_ood_idx].tolist()) == {main_ood_center}, "test_ood contient un autre centre"
    for k in ["train", "val", "calib", "test_id"]:
        assert main_ood_center not in set(centers[splits[k]].tolist()), f"Fuite du centre OOD dans {k}"

    return splits


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def print_summary(splits, labels, n_classes=8):
    labels = np.asarray(labels)
    print("\n--- Tailles des splits ---")
    for k, v in splits.items():
        print(f"{k:15s}: {len(v)}")

    print("\n--- Distribution des classes par split (%) ---")
    header = "split".ljust(15) + "".join(f"c{c}".rjust(7) for c in range(n_classes))
    print(header)
    for k, v in splits.items():
        cnt = Counter(labels[v].tolist())
        denom = len(v) if len(v) > 0 else 1
        row = k.ljust(15) + "".join(f"{100.0 * cnt.get(c, 0) / denom:7.1f}" for c in range(n_classes))
        print(row)

    print("\n--- Effectifs par classe (alerte si < 30) ---")
    for k in ["calib", "test_id", "ood_calib_pool", "test_ood"]:
        cnt = Counter(labels[splits[k]].tolist())
        small = {c: cnt.get(c, 0) for c in range(n_classes) if cnt.get(c, 0) < 30}
        if small:
            print(f"{k:15s}: classes avec peu d'echantillons -> {small}")


def save_splits(splits, path):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: np.asarray(v, dtype=np.int64).tolist() for k, v in splits.items()}, f)
    print(f"Splits sauvegardes : {out_path}")


def load_splits(path):
    with open(path, "r", encoding="utf-8") as f:
        return {k: np.array(v, dtype=np.int64) for k, v in json.load(f).items()}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--main_ood_center", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # NB : on utilise uniquement le split HF "train" (18597 images).
    # Le split HF "test" (4650 images) n'est PAS utilise dans ce protocole.
    REVISION = "d1457caea44f57dfe855bcf67e2ec09bb06781e9"   

    dataset = load_dataset("flwrlabs/fed-isic2019", split="train", revision=REVISION)  
    centers = np.array(dataset["center"])
    labels = np.array(dataset["label"])

    print("Total images :", len(dataset))
    print("Par centre   :", dict(sorted(Counter(centers.tolist()).items())))
    print("Par classe   :", dict(sorted(Counter(labels.tolist()).items())))

    splits = make_splits(centers, labels, args.main_ood_center, args.seed)
    print_summary(splits, labels)
    print("\nZero chevauchement verifie : OK")

    out = args.out or f"data/splits/splits_ood{args.main_ood_center}_seed{args.seed}.json"
    save_splits(splits, out)