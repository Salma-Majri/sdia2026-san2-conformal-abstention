# SAN-2 : Incertitude calibrée et abstention en aide au diagnostic

Projet de fin de module, IA avancée (Master SDIA, ENSET Mohammedia, 2026-2027).

## Question de recherche

La prédiction conforme tient-elle sa garantie de couverture sur une tâche de diagnostic
lorsque la distribution de test change (nouveau centre d'acquisition, sous-population rare),
et quel taux d'abstention faut-il payer pour la rétablir ?

## Données

FED-ISIC 2019 (dermoscopie, 8 classes, 6 centres). Les images ne sont pas redistribuées
dans ce dépôt (licence ISIC) : elles sont téléchargées par `data/download.py`.

## Structure du dépôt

```
configs/protocol.yaml     protocole figé
data/download.py          téléchargement des données
data/splits/              indices des splits (versionnés)
src/                      code source (splits, dataset, entraînement, conforme...)
scripts/                  pilote du bruit, figures, relance de l'expérience
notebooks/                analyse des erreurs
docs/                     notes et protocole
outputs/                  logits, résultats, figures (non versionnés sauf results/figures)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduire

À compléter : une commande unique relancera l'expérience principale (`bash run_all.sh`).

## État d'avancement

- [x] Structure du projet
- [ ] Splits
- [ ] Protocole figé
- [ ] Dataset et entraînement (seeds 0, 1, 2)
- [ ] Extraction des logits
- [ ] Pilote du bruit (sigma) sur val
- [ ] Métriques et temperature scaling
- [ ] Prédiction conforme (LAC, APS, marginale, par classe)
- [ ] Expérience principale : ID, bruit d'acquisition, nouveau centre
- [ ] Décalage : sous-population rare
- [ ] Décalage : déséquilibre de prévalence
- [ ] Article

## Équipe

- Salma Majri
- Fatima zahra Massane

Encadrant : Pr. Soufiane HAMIDA.