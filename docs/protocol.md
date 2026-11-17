# Protocole expérimental SAN-2

**Version** : v1.0 (`sigma_levels` en attente du pilote)
**Statut** : figé sauf `sigma_levels`, à remplir après le pilote sur val (passage en v1.1)
**Source de vérité** : `configs/protocol.yaml`. Ce document l'explique ; en cas de divergence, le YAML fait foi.

## 1. Question de recherche

La prédiction conforme tient-elle sa garantie de couverture sur une tâche de diagnostic dermatologique lorsque la distribution de test change (nouveau centre d'acquisition, sous-population rare, bruit d'acquisition, déséquilibre de prévalence), et quel taux d'abstention faut-il payer pour la rétablir ?

Hypothèse nulle de référence : sans décalage, la couverture empirique est proche de 1 - alpha. Avec décalage, elle peut tomber sous 1 - alpha, ce que l'expérience mesure.

## 2. Données

- Jeu : FED-ISIC 2019 (`flwrlabs/fed-isic2019`), 18 597 images, 8 classes, 6 centres.
- Seul le split HF `train` est utilisé. Le split HF `test` n'est pas utilisé.
- Les images ne sont pas redistribuées (licence ISIC) : téléchargement par `data/download.py`, révision HF figée.
- L'entraînement est centralisé. Le caractère multi-centres du jeu sert à créer un décalage de domaine réel, pas à faire de l'apprentissage fédéré.

## 3. Splits

Centre OOD principal : **3**, jamais vu à l'entraînement. Seed des splits : 0 (fixe pour tous les runs). Fichier : `data/splits/splits_ood3_seed0.json`.

| Split | Origine | Taille | Usage |
|---|---|---|---|
| train | centres 0, 1, 2, 4, 5 | 10 913 | entraînement |
| val | centres sources | 1 679 | early stopping, réglages, temperature scaling, pilote du bruit |
| calib | centres sources | 2 518 | calibration conforme ID |
| test_id | centres sources | 1 680 | évaluation ID, utilisé une seule fois |
| ood_calib_pool | centre 3 | 361 | calibration conforme avec peu d'exemples OOD |
| test_ood | centre 3 | 1 446 | évaluation OOD, utilisé une seule fois |

Proportions : sources 65 / 10 / 15 / 10 % (train / val / calib / test_id) ; centre 3 : 20 % pool de calibration, 80 % test. Stratification par classe sur le centre 3, par centre × classe pour le premier split source (strates rares fusionnées), par classe pour les suivants.

**Règle d'usage** : aucun réglage d'hyperparamètre sur test_id ni test_ood. Chacun n'est lu qu'une fois.

## 4. Modèle et entraînement

- Backbone : ResNet-50 pré-entraîné ImageNet, dropout 0.3 avant la couche finale (réutilisé pour MC-Dropout).
- Images : 224 × 224, normalisation ImageNet.
- AdamW, lr 3e-4, weight decay 1e-4, batch 64, 30 époques maximum, scheduler cosine, cross-entropy.
- Early stopping sur val_loss, patience 5. Précision mixte.
- Pas de pondération des classes dans la baseline (variante en ablation).
- Augmentations d'entraînement : random resized crop, flips horizontal et vertical, color jitter léger. Évaluation : resize, center crop.
- Seeds d'entraînement : 0, 1, 2. Résultats rapportés en moyenne ± écart-type.
- Déterminisme : `torch.manual_seed`, seed numpy, `cudnn.deterministic=True`. Matériel : Colab T4.

## 5. Méthodes comparées

1. **Prédiction conforme fractionnée** : scores LAC et APS (non randomisé), marginale et par classe. Repli sur le quantile global si une classe a moins de 20 exemples de calibration.
2. **MC-Dropout** : 20 passes.
3. **Ensemble** : 3 modèles, un par seed d'entraînement.

Temperature scaling ajusté sur val, un T par seed. Niveau principal alpha = 0.10 ; sensibilité alpha ∈ {0.05, 0.20}.

Sources de calibration : `calib` (ID), `ood_calib_pool` (OOD few-shot), mixte (union simple, sans pondération).

Abstention : le modèle s'abstient lorsque la taille de l'ensemble de prédiction est différente de 1. Grille de 50 seuils pour la courbe risque-couverture.

## 6. Décalages simulés

Le centre 3 est déjà un décalage réel. Trois décalages contrôlés s'y ajoutent, tous calculés à partir du même run final, sans réentraînement ni recalibration.

### 6.1 Bruit d'acquisition
- Bruit gaussien appliqué à `test_id`, écart-type σ exprimé sur l'image en [0, 1], avant normalisation ImageNet.
- Bruit déterministe (graine dérivée de l'indice de l'image), identique pour toutes les méthodes.
- **Pilote sur val** avec le modèle de seed 0, grille σ ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.08}.
- Règle : pour chaque niveau (léger, moyen, fort), choisir le σ de la grille dont la chute d'accuracy sur val est la plus proche de 2, 5 et 10 points.
- `sigma_levels` est rempli et commité (v1.1) **avant** toute lecture de test_id.

### 6.2 Sous-population rare
- Évaluation conditionnelle au sous-groupe sur `test_id`, sans rééchantillonnage, calibration inchangée.
- Par centre (0, 1, 2, 4, 5 ; centres rares : 4 et 5) et par classe rare (3, 5, 6, 7).
- Métriques : couverture, taille moyenne des ensembles, taux de singletons.
- Moins de 30 images dans un sous-groupe : résultat marqué « indicatif ».

### 6.3 Déséquilibre de prévalence
- Rééchantillonnage pondéré avec remise de `test_id`, 200 tirages de 1 680 images, graine du tirage i = i.
- Scénarios : prévalence uniforme ; inverse de la prévalence dans calib ; classes rares (3, 5, 6, 7) pondérées ×5 puis renormalisées.
- Calibration inchangée (calib ID), marginal et par classe comparés.
- Rapport : moyenne et écart-type sur les tirages et sur les seeds.

## 7. Métriques

Couverture empirique, taille moyenne des ensembles, taux de singletons, risque sélectif, courbe risque-couverture, ECE, couverture par classe, accuracy et F1 macro.

## 8. Ablations

- Sans temperature scaling
- LAC vs APS
- Marginal vs par classe
- Calibration ID vs OOD vs mixte
- Taille du pool OOD : 0, 50, 100, 361
- Avec vs sans pondération des classes

## 9. Limites connues

- Classes rares (5 et 6) : très peu d'échantillons dans calib, test_id et test_ood. Sur test_ood, la classe 6 n'a que 2 images : sa couverture n'est pas interprétable.
- `ood_calib_pool` : classe 6 = 1 image, classe 5 = 4, classe 3 = 17. La calibration par classe y est impossible pour ces classes (repli sur le quantile global), et le repli s'étend quand le pool est sous-échantillonné.
- APS non randomisé : couverture garantie mais légèrement conservatrice.
- Calibration mixte : mélange simple, sans pondération.
- Rééchantillonnage de prévalence avec remise : les classes rares de test_id sont dupliquées, variance élevée.
- Sous-population par centre : les centres 4 et 5 sont peu représentés dans test_id, intervalles larges.
- Un seul centre OOD principal (le 3) : les conclusions sur le « nouveau centre » ne sont pas généralisables à tous les centres.

## 10. Historique des versions

| Version | Tag git | Contenu |
|---|---|---|
| v1.0 | `protocol-v1.0` | Protocole complet, révision HF fixée, `sigma_levels` à `null` |
| v1.1 | `protocol-v1.1` | `sigma_levels` fixés après le pilote sur val, avant toute évaluation test |