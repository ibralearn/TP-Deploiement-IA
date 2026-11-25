# Chaînes de déploiement d'un modèle d'IA — TP n°1

Comparaison de **quatre pipelines d'inférence** d'un même modèle PyTorch (ResNet-18 pré-entraîné ImageNet, tête 10 classes) sur **CPU** et **GPU** dans Google Colab, en suivant l'enchaînement *haut niveau → format du modèle → middle level → bas niveau → matériel*.

| | Pipeline | Format | Middle level | Bas niveau | Matériel |
|---|---|---|---|---|---|
| **CPU n°1** | PyTorch → ONNX → OpenVINO | ONNX (opset 17) | OpenVINO Runtime 2026.4 (+ NNCF INT8) | oneDNN (AVX2) | Colab CPU, Xeon 2,2 GHz, 2 vCPU |
| **CPU n°2** | PyTorch → TorchScript / torch.compile | TorchScript, graphe FX | freeze + optimize_for_inference, Inductor (C++) | oneDNN, code C++ généré | Colab CPU |
| **GPU n°1** | PyTorch → ONNX → TensorRT | ONNX (même fichier) | TensorRT 11.5 (FP32, FP16) | CUDA 13.0, cuDNN 9.28 | Colab GPU, Tesla T4 |
| **GPU n°2** | PyTorch → TorchScript / torch.compile | TorchScript, graphe FX | Inductor (Triton), CUDA Graphs (FP32, FP16) | CUDA, cuDNN, kernels Triton | Colab GPU, Tesla T4 |

## Organisation du dépôt

```
├── README.md
├── rapport/
│   ├── Rapport_TP_Deploiement_IA.pdf   ← rapport technique (protocole, résultats, graphiques, analyse)
│   └── rapport.html                    ← source du rapport
├── notebooks/                          ← les 4 notebooks Colab exécutés, un par pipeline
│   ├── Pipeline_CPU1_PyTorch_ONNX_OpenVINO.ipynb
│   ├── Pipeline_CPU2_PyTorch_TorchScript_TorchCompile.ipynb
│   ├── Pipeline_GPU1_PyTorch_ONNX_TensorRT.ipynb
│   └── Pipeline_GPU2_PyTorch_TorchScript_TorchCompile.ipynb
├── figures/                            ← graphiques de comparaison des 4 pipelines
├── resultats/
│   ├── resultats.json                  ← mesures de synthèse utilisées dans le rapport et les figures
│   └── statistiques_detaillees.json    ← moyenne, écart-type, min et max de chaque indicateur
└── scripts/figures.py                  ← régénère les figures à partir de resultats.json
```

## Protocole (identique pour les 4 notebooks)

- **Modèle** : ResNet-18 torchvision (`IMAGENET1K_V1`), dernière couche remplacée par une couche 512 → 10 apprise par *linear probing* (réseau gelé, 3 000 images d'entraînement CIFAR-10, graine 42).
- **Données** : 50 images de test CIFAR-10 (5 par classe, graine fixe), mêmes indices et même empreinte SHA-256 dans les 4 notebooks.
- **Prétraitement** : uint8 → [0, 1], redimensionnement bilinéaire 32×32 → 224×224, normalisation ImageNet (moyenne 0,485 / 0,456 / 0,406 ; écart-type 0,229 / 0,224 / 0,225).
- **Mesures** : batch de 10, 5 passes de chauffe, **10 passes mesurées** (moyenne, écart-type, min, max, coefficient de variation), Top-1 et accord des prédictions avec PyTorch eager, **3 mesures d'énergie** de 10 s (GPU : NVML, mesurée ; CPU : estimée, RAPL indisponible sur Colab).
- Chaque notebook mesure aussi une **référence PyTorch eager FP32 dans la même session**, qui sert de base aux accélérations.

## Exécution

1. Ouvrir un notebook dans Google Colab (*Fichier → Importer le notebook* ou depuis GitHub).
2. Choisir le type d'exécution : **CPU** pour les notebooks CPU, **GPU T4** pour les notebooks GPU (*Exécution → Modifier le type d'exécution*).
3. *Exécution → Tout exécuter* et autoriser l'accès à Google Drive. Le premier notebook lancé crée les fichiers communs (images, poids, ONNX) dans `MyDrive/TP_Deploiement_IA/shared` ; les suivants les rechargent, ce qui garantit le même modèle et les mêmes entrées.
4. La section « Comparaison des pipelines » de chaque notebook rassemble les résultats présents dans `MyDrive/TP_Deploiement_IA/results`.

Durée : environ 10 à 15 minutes par notebook GPU, davantage pour les notebooks CPU. Le téléchargement de CIFAR-10 depuis le serveur officiel peut être très lent sur Colab ; les notebooks GPU contiennent une cellule qui le télécharge depuis un miroir et vérifie son empreinte MD5 officielle.

Pour régénérer les figures du rapport : `python scripts/figures.py` (matplotlib).

## Principaux résultats

| Variante | Débit (img/s) | Accélération* | Top-1 | Énergie / image |
|---|---:|---:|---:|---:|
| CPU — PyTorch eager FP32 | 11,0 – 12,6 | ×1,00 | 90 % | 1 623 – 1 896 mJ (estimée) |
| CPU n°1 — OpenVINO FP32 | 13,6 ± 2,3 | ×1,24 | 90 % | 1 039 mJ (estimée) |
| CPU n°1 — OpenVINO INT8 | 27,6 ± 4,2 | ×2,52 | 92 % | 558 mJ (estimée) |
| CPU n°2 — torch.compile + freezing FP32 | 18,6 ± 3,3 | ×1,48 | 90 % | 1 196 mJ (estimée) |
| GPU — PyTorch eager FP32 | 980 – 1 020 | ×1,00 | 90 % | 69 – 72 mJ |
| GPU n°1 — TensorRT FP32 | 1 235 ± 15 | ×1,21 | 90 % | 55,8 mJ |
| GPU n°1 — **TensorRT FP16** | **4 272 ± 111** | **×4,19** | 90 % | **15,9 mJ** |
| GPU n°2 — TorchScript FP32 | 1 152 ± 9 | ×1,18 | 90 % | 60,6 mJ |
| GPU n°2 — torch.compile FP32 | 880 ± 9 | ×0,90 | 90 % | 78,6 mJ |
| GPU n°2 — torch.compile + CUDA Graphs FP16 | 2 923 ± 663 | ×2,98 | 90 % | 20,0 mJ |

\* par rapport à la référence PyTorch eager mesurée dans la même session. Le rapport donne, pour chacune des 16 variantes et pour chaque indicateur (débit, latence, temps des 50 images, Top-1, énergie, puissance), la moyenne, l'écart-type, le minimum et le maximum (section 4.2).

## Conclusions

- **Le matériel fixe l'ordre de grandeur** : à code égal, le GPU T4 est 78 à 93 fois plus rapide que les 2 vCPU de Colab.
- **En FP32, les optimisations de graphe apportent peu** (×0,9 à ×1,5) : ResNet-18 est dominé par les convolutions, déjà confiées à oneDNN (CPU) ou cuDNN (GPU) par tous les moteurs. La fusion d'opérateurs (49 nœuds ONNX → 25 couches avec OpenVINO et TensorRT) supprime surtout BatchNorm, ReLU et additions.
- **La précision réduite est le levier principal** : ×2,5 en INT8 avec OpenVINO, ×4,2 en FP16 avec TensorRT, avec une énergie par image réduite d'autant (−77 % pour TensorRT FP16).
- **Les prédictions ne sont pas dégradées** : toutes les variantes FP32 et FP16 donnent exactement les mêmes 50 prédictions que PyTorch ; seule l'INT8 en modifie 2 sur 50.
- **Les mesures GPU sont stables et reproductibles** (CV ≈ 1 % en FP32, moins de 2 % d'écart entre deux sessions indépendantes) ; **les mesures CPU sur Colab sont très variables** (CV ≈ 20 %), ce qui empêche de départager les variantes CPU FP32.
- Recommandation : **TensorRT FP16** sur GPU ; **OpenVINO INT8** sur CPU si une légère variation des prédictions est acceptable.
