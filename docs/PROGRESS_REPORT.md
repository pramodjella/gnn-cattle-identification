# Progress Report
## Biometric Cattle Identification via Deep Learning and Graph Neural Networks

**Student:** Pramod Jella
**Date:** 23 June 2026
**Status:** Core experiments complete; manuscript drafted; deployment demonstrator built.

---

## 1. Summary for the Mentor

The end-to-end pipeline is **complete and reproducible**, all four model families plus two prior-art baselines are **trained and evaluated** on the 964-image test set, results are **statistically validated**, and a **manuscript draft + publication figures** are ready. A working **web demonstrator** (register / identify cattle) has also been built.

**Headline result:** the proposed **Ensemble (CNN + Hybrid CNN-GNN)** reaches **96.1% Rank-1** accuracy with a **0.78% Equal Error Rate** and **0.9995 ROC AUC**, outperforming re-implemented VGG-16 (95.1%) and ResNet-50 (94.6%) prior-art baselines.

The main open items are honest ones, called out in §6: pure GNNs trail CNNs on raw accuracy, and the Hybrid model is unstable under short cross-validation training. These are the points I'd most like to discuss.

## 2. What Is Complete

| Component | Status | Where |
| :-- | :-- | :-- |
| Data download + 70/15/15 split | ✅ | `scripts/01_download_data.py` |
| Preprocessing (ROI, CLAHE, Otsu seg.) | ✅ | `scripts/02_preprocess.py`, `src/preprocessing/` |
| Keypoint extraction (DISK + alternates) | ✅ | `scripts/03_extract_keypoints.py`, `src/features/` |
| k-NN graph construction (k=8) | ✅ | `scripts/04_build_graphs.py`, `src/features/graph_builder.py` |
| CNN (EfficientNet-B4 + ArcFace) | ✅ trained | `scripts/train_cnn.py`, `src/models/cnn_model.py` |
| Hybrid CNN-GNN | ✅ trained | `scripts/train_hybrid.py`, `src/models/hybrid_model.py` |
| ProtoN (Prototype Node GNN) | ✅ trained | `scripts/train_proton.py`, `src/models/proton.py` |
| GNN v3 / v4 (GATv2 + Virtual Node) | ✅ trained | `scripts/train_gnn_v3*.py`, `src/models/gnn_v3.py` |
| GNN+ / GNN++ ablation variants | ✅ trained | `scripts/train_gnn_plus*.py` |
| VGG-16 baseline (Bello et al. 2020) | ✅ trained | `scripts/baselines/train_vgg_baseline.py` |
| ResNet-50 baseline (Qin et al. 2021) | ✅ trained | `scripts/baselines/train_resnet_baseline.py` |
| Ensemble + Test-Time Augmentation | ✅ | `scripts/ensemble_inference.py`, `scripts/evaluate_with_tta.py` |
| 5-fold cross-validation | ✅ | `scripts/cross_validation.py` |
| McNemar tests + bootstrap CIs | ✅ | `scripts/statistical_tests.py` |
| Explainability (Grad-CAM + GNN attention) | ✅ | `scripts/visualize_gradcam.py`, `scripts/visualize_gnn_attention.py` |
| Publication figures (PDF + PNG) | ✅ | `scripts/figures/generate_paper_figures.py` |
| LaTeX tables | ✅ | `outputs/stats/*.tex` |
| Manuscript draft | ✅ | [`paper_draft.md`](../paper_draft.md) |
| Web demonstrator (FastAPI + React + pgvector) | ✅ built | `web/` |

## 3. Results (test set, 964 images)

| Model | Rank-1 (%) | Rank-5 (%) | EER (%) | ROC AUC | Notes |
| :-- | :-: | :-: | :-: | :-: | :-- |
| **Ensemble (CNN-TTA + Hybrid)** | **96.1** | **98.1** | **0.78** | **0.9995** | Proposed SOTA (weights CNN 0.95 / Hybrid 0.05) |
| CNN (EfficientNet-B4) + TTA | 95.4 | 97.4 | 2.70 | 0.9961 | ArcFace + test-time augmentation |
| CNN (EfficientNet-B4) | 95.4 | 97.4 | 2.70 | 0.9961 | Baseline ArcFace |
| VGG-16 baseline | 95.1 | 97.7 | 1.23 | 0.9993 | Bello et al. (2020) re-impl. |
| ResNet-50 baseline | 94.6 | 97.3 | 2.14 | 0.9971 | Qin et al. (2021) re-impl. |
| **Hybrid CNN-GNN** | 92.0 | 96.7 | 1.85 | 0.9979 | Proposed: bilinear sampling + EdgeConv + TRM |
| ProtoN (Prototype GNN) | 91.6 | 94.8 | 1.17 | 0.9982 | Proposed: cross-graph alignment loss |
| GNN v4 (GATv2, enhanced) | 91.6 | 94.4 | 1.48 | 0.9937 | 4-layer GATv2 + virtual node |
| GNN v3 (GATv2 + VN) | 91.5 | 95.0 | 1.87 | 0.9954 | 4-layer GATv2 + virtual node |
| GNN++ (CNN patch features) | 78.3 | 86.2 | 7.81 | 0.9730 | MobileNetV3 patch nodes |
| GNN+ (Kornia-DISK features) | 72.0 | 84.2 | 11.17 | 0.9516 | DISK-descriptor nodes |

## 4. Key Findings

1. **CNN texture is the strongest single signal.** Pure CNNs (≈95% Rank-1) beat pure GNNs (≈91–92%), indicating raw dermatoglyphic texture (groove width, bead density) is more discriminative than keypoint geometry alone.
2. **Topology helps verification.** The Hybrid and ProtoN GNNs achieve the **lowest EERs** (1.85% and 1.17%), so graph structure improves the accept/reject reliability even where it doesn't top raw Rank-1.
3. **The ensemble is the best of both.** Blending the CNN's global texture view with the Hybrid model's topological view yields the best overall numbers (96.1% Rank-1, 0.78% EER) — the two error profiles are complementary.
4. **Learned keypoints beat handcrafted ones.** DISK-based `GNN+` (72.0%) clearly outperforms SIFT-equivalent baselines, validating the move to learned descriptors.
5. **Results are statistically significant.** McNemar tests confirm the CNN/Hybrid advantage over the pure GNNs at *p* < 10⁻⁸.

## 5. Validation & Rigour

- **Stratified 5-fold cross-validation** on the top models:
  - CNN (EfficientNet-B4): **93.91% ± 0.31%** Rank-1 — very stable.
  - ProtoN: **89.49% ± 0.71%** Rank-1 — stable.
  - Hybrid: **68.88% ± 2.04%** Rank-1 — unstable under CV-safe short training (see §6).
- **McNemar pairwise significance tests** and **bootstrap confidence intervals** computed for all headline comparisons.
- **Explainability** confirms models attend to anatomy: Grad-CAM concentrates on central bead clusters; GNN attention emphasises keypoint links spanning major valleys.

## 6. Open Issues / Points to Discuss With You

1. **GNNs don't beat CNNs on raw accuracy.** This is the honest core tension. The research contribution is currently best framed as *"topology improves verification reliability and ensembles, and learned-keypoint graphs are a viable representation"* rather than *"GNNs win outright."* I'd value your view on how to frame/strengthen this.
2. **Hybrid cross-validation instability (68.9% ± 2.0%).** Full-protocol Hybrid scores 92% Rank-1, but under the shorter, CV-safe training schedule it collapses. Likely a training-budget / optimisation-stability issue rather than a modelling flaw — needs longer CV runs or a more robust schedule to confirm.
3. **Validation vs. test gap.** Reported "best validation Rank-1" (~82–84%) sits *below* test Rank-1 (~95%), an artefact of the small validation set (615 images, ~2.4/animal) and the gallery/probe matching protocol. Worth aligning the val/test evaluation protocol before submission.
4. **Single dataset.** All results are on one benchmark; cross-dataset generalisation is untested.

## 7. Next Steps

**Near term (pre-submission)**
- Re-run Hybrid cross-validation with a longer/robust schedule to resolve the instability (§6.2).
- Reconcile the val/test evaluation protocol (§6.3).
- Finalise manuscript figures/tables and tighten the framing of the GNN contribution (§6.1).

**Medium term**
- Cross-dataset transfer test (train on Zenodo, evaluate on a second muzzle dataset) for generalisation evidence.
- Robustness study: performance under simulated occlusion / blur / low light — the regime where topology should pay off most.
- Mobile/low-latency deployment of the best model in the web demonstrator for a field-realistic demo.

**Stretch**
- Submit to *Computers and Electronics in Agriculture*.

## 8. Reproduce Key Results

```powershell
# 5-fold cross-validation (top 3 models)
venv\Scripts\python.exe scripts/cross_validation.py --epochs-cnn 10 --epochs-proton 12 --epochs-hybrid-p1 10 --epochs-hybrid-p2 2

# Comparison report + markdown table
venv\Scripts\python.exe scripts/compare_models.py

# Publication-quality vector figures (PDF/PNG)
venv\Scripts\python.exe scripts/figures/generate_paper_figures.py

# Explainability maps
venv\Scripts\python.exe scripts/visualize_gradcam.py
venv\Scripts\python.exe scripts/visualize_gnn_attention.py
```

---

*Companion document: [PROJECT_INTRODUCTION.md](PROJECT_INTRODUCTION.md) — motivation, approach, and scope.*
