# SAN-2 : Incertitude calibrée et abstention en aide au diagnostic

Projet de fin de module, IA avancée (Master SDIA, ENSET Mohammedia, 2026-2027).

## Question de recherche

La prédiction conforme tient-elle sa garantie de couverture sur une tâche de diagnostic
lorsque la distribution de test change (nouveau centre d'acquisition, bruit d'acquisition,
sous-population rare, déséquilibre de prévalence), et quel taux d'abstention faut-il payer
pour la rétablir ?

## Données

FED-ISIC 2019 (`flwrlabs/fed-isic2019`, Hugging Face), 18 597 images dermoscopiques,
8 classes, 6 centres. Seul le split HF `train` est utilisé. Les images ne sont pas
redistribuées dans ce dépôt (licence ISIC) : elles sont téléchargées par script, à une
révision figée dans `configs/protocol.yaml`.

Centre OOD principal : **3** (jamais vu à l'entraînement). L'entraînement est centralisé :
le caractère multi-centres du jeu sert à créer un décalage de domaine réel.

| Split | Origine | Taille | Usage |
|---|---|---|---|
| train | centres 0, 1, 2, 4, 5 | 10 913 | entraînement |
| val | centres sources | 1 679 | early stopping, réglages, pilote du bruit |
| calib | centres sources | 2 518 | calibration conforme ID |
| test_id | centres sources | 1 680 | évaluation ID (une seule fois) |
| ood_calib_pool | centre 3 | 361 | calibration conforme OOD |
| test_ood | centre 3 | 1 446 | évaluation OOD (une seule fois) |

## Protocole

Figé avant les résultats : voir [`configs/protocol.yaml`](configs/protocol.yaml) et
[`docs/protocol.md`](docs/protocol.md). Version courante : **v1.1** (tag `protocol-v1.1`),
`sigma_levels` fixés après le pilote sur val. Modèle : ResNet-50 pré-entraîné ImageNet,
seeds d'entraînement 0, 1, 2, seed des splits 0.

## Structure du dépôt

```
configs/protocol.yaml        protocole figé
docs/protocol.md             protocole expliqué
data/download.py             téléchargement des données
data/splits/                 indices des splits (JSON versionnés)
src/splits.py                génération des splits
src/utils.py                 config, seeds, chargement du dataset, des splits, du modèle
src/dataset.py               Dataset PyTorch (bruit gaussien optionnel, déterministe par image)
src/train.py                 entraînement
src/extract_logits.py        extraction des logits par split
src/pilot_noise.py           pilote du bruit sur val
src/metrics.py               métriques (accuracy, F1, NLL, ECE) et temperature scaling (val)
src/conformal.py             prédiction conforme : LAC, APS, marginale / par classe, sources de calibration
src/shifts.py                décalages : sous-population rare, déséquilibre de prévalence
src/run_conformal.py         expériences conformes (modes smoke et final)
scripts/make_figures.py      figures
notebooks/                   analyse des erreurs
checkpoints/                 modèles entraînés (non versionnés)
outputs/                     logits, figures (non versionnés) ; résultats versionnés : pilote, métriques, final_conformal.json
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduire

Étapes déjà disponibles (GPU recommandé pour l'entraînement) :

```bash
python src/splits.py --main_ood_center 3 --seed 0 --out data/splits/splits_ood3_seed0.json
python src/train.py --seed 0                    # puis --seed 1 et --seed 2
python src/extract_logits.py --seed 0           # puis --seed 1 et --seed 2
python src/pilot_noise.py --seed 0                 # pilote du bruit sur val (seed 0)
python src/extract_logits.py --seed 0 --splits test_id --noise_sigma 0.01   # puis 0.0125, 0.02 ; seeds 1 et 2
```

Le pilote a sélectionné les sigmas suivants : léger=0.01, moyen=0.0125, fort=0.02 (voir outputs/results/pilot_noise_seed0.json)

Métriques et temperature scaling (ajusté sur val, un T par seed) :

```bash
python src/metrics.py
```

Prédiction conforme. Le mode `smoke` calibre sur `calib` et évalue sur `val` (aucun jeu de test lu) ;
le mode `final` lit `test_id` et `test_ood` **une seule fois** (il refuse de tourner si le fichier de
sortie existe, si l'arbre git n'est pas propre ou si `sigma_levels` est null) :

```bash
python src/run_conformal.py --mode smoke
python src/run_conformal.py --mode final        # une seule exécution : résultats dans outputs/results/final_conformal.json
```

Le run final calcule dans la même exécution : ID, bruit d'acquisition, nouveau centre (3 sources de
calibration, alpha 0.10 / 0.05 / 0.20), puis les décalages de sous-population et de prévalence sur `test_id`.

Les checkpoints et les logits ne sont pas versionnés : ils se régénèrent avec ces commandes.
L'entraînement a été exécuté sur Google Colab (GPU T4).

Expérience complète (commande unique, à venir) :

```bash
bash run_all.sh
```

## État d'avancement

- [x] Structure du projet
- [x] Splits
- [x] Protocole figé (sigma_levels fixés après le pilote)
- [x] Dataset et entraînement (seeds 0, 1, 2)
- [x] Extraction des logits (sans bruit)
- [x] Pilote du bruit (sigma) sur val
- [x] Extraction des logits avec bruit (test_id, sigmas figés)
- [x] Métriques
- [x] Temperature scaling
- [x] Prédiction conforme (LAC, APS, marginale, par classe)
- [x] Expérience principale : ID, bruit d'acquisition, nouveau centre
- [x] Décalage : sous-population rare
- [x] Décalage : déséquilibre de prévalence
- [ ] MC-Dropout
- [ ] Ensembles
- [ ] Ablations
- [ ] Analyse des erreurs et figures
- [ ] Article

## Équipe

- Salma Majri
- Fatima zahra Massane

Encadrant : Pr. Soufiane HAMIDA.