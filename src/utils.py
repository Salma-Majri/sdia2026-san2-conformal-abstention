"""Utilitaires communs : config, seeds, device, dataset, splits, modele."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path="configs/protocol.yaml"):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_split_indices(cfg):
    """Retourne {nom_split: np.array d'indices} depuis le JSON fige par splits.py."""
    p = ROOT / cfg["splits"]["file"]
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {k: np.array(v, dtype=np.int64) for k, v in d.items()}


def load_hf_dataset(cfg):
    """Charge FedISIC2019 a la revision figee dans protocol.yaml."""
    from datasets import load_dataset

    d = cfg["dataset"]
    revision = d.get("revision")
    if not revision:
        raise ValueError("dataset.revision est null dans protocol.yaml : fige-le avant de continuer.")
    return load_dataset(d["name"], split=d["hf_split"], revision=revision)


def build_model(cfg, pretrained=None):
    """ResNet-50 + dropout avant la couche finale (sert aussi au MC-Dropout).

    pretrained=None : suit le protocole. Mettre False pour recharger un checkpoint.
    """
    from torchvision import models

    m = cfg["model"]
    if pretrained is None:
        pretrained = m["pretrained"] == "imagenet"
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet50(weights=weights)
    net.fc = nn.Sequential(nn.Dropout(m["dropout"]),
                           nn.Linear(net.fc.in_features, cfg["dataset"]["n_classes"]))
    return net