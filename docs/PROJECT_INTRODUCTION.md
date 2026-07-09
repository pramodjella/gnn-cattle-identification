# Project Introduction
## Biometric Cattle Identification via Deep Learning and Graph Neural Networks

**Student:** Pramod Jella
**Date:** 23 June 2026
**Target venue:** *Computers and Electronics in Agriculture* (research article)

---

## 1. Motivation

Livestock traceability underpins disease control, food-safety assurance, insurance and ownership verification, and modern herd management. The methods used in practice today — ear tags, hot/freeze branding, and RFID transponders — are all **invasive, removable, and forgeable**. Tags fall out or are swapped; brands fade and cause welfare concerns; RFID chips can be cloned or transferred between animals.

The **muzzle print** (nose print) offers a biometric alternative. Like a human fingerprint, the arrangement of *beads* (raised dermal protuberances) and *valleys* (grooves) on a bovine muzzle is unique to each animal and stable over its lifetime. It cannot be removed or transferred, and it can be captured with an ordinary camera. This makes muzzle-based identification an attractive, low-cost, tamper-proof, and non-invasive option for the field.

The technical challenge is that muzzle images captured in real conditions are hard:

1. **Geometric deformation** — head movement and camera angle change the apparent spacing of beads.
2. **Environmental variation** — outdoor lighting, shadows, dirt, and moisture alter contrast.
3. **Scale change** — the muzzle grows as the animal matures, so matching must be scale-invariant.

Classical handcrafted descriptors (SIFT, LBP) generalise poorly under these conditions, and even standard CNNs — which are translation-equivariant but not naturally invariant to non-rigid deformation — leave room for improvement.

## 2. Core Idea

This project investigates whether modelling the muzzle as a **graph of keypoints** — rather than only as a grid of pixels — improves identification robustness, and how graph-based representations compare against strong CNN baselines on a level playing field.

The central hypothesis: the *topology* of the muzzle (which beads neighbour which, and how they are spatially arranged) carries identity information that is more robust to occlusion and deformation than appearance alone. Graph Neural Networks (GNNs) can learn over this topology via message passing, capturing relational structure ("bead A always sits between beads B and C") that a CNN does not explicitly represent.

Rather than betting on a single architecture, the project is framed as a **rigorous, controlled comparative study** of four model families on one benchmark, so that any claimed advantage is measured fairly and tested for statistical significance.

## 3. Objectives

1. Build a complete, reproducible pipeline from raw muzzle images to identity decisions.
2. Implement and tune representatives of four model families:
   - **Pure CNN** (appearance/texture only),
   - **Pure GNN** (keypoint topology only),
   - **Hybrid CNN-GNN** (texture sampled at keypoints, fused with topology),
   - **Prototype GNN** (metric-learning over graphs).
3. Re-implement **prior-art baselines** (VGG-16, ResNet-50) for fair comparison against published methods.
4. Evaluate with **biometric metrics** (Rank-1/Rank-5 identification, Equal Error Rate, ROC AUC), not just classification accuracy.
5. Establish **statistical rigour** via stratified 5-fold cross-validation, McNemar significance tests, and bootstrap confidence intervals.
6. Provide **explainability** (Grad-CAM for CNNs, attention heatmaps for GNNs) to verify models attend to biologically meaningful structures.
7. Package the work as a **publication-ready manuscript** with figures/tables, and a **deployable web application** demonstrating real-world use.

## 4. Approach (Pipeline Overview)

```
Raw muzzle image
   │
   ▼  (1) Preprocessing
ROI extraction → CLAHE contrast enhancement → Otsu segmentation (256×256)
   │
   ▼  (2) Keypoint detection & description
Learned keypoints (Kornia-DISK; SuperPoint / DeDoDe / SIFT also supported)
→ up to 128 keypoints, each with a 256-d descriptor
   │
   ▼  (3) Graph construction
k-NN graph (k=8) over keypoint coordinates
edges carry 5-d features [Δx, Δy, distance, angle, relative scale]
   │
   ▼  (4) Model family (one of four)
CNN  |  GNN  |  Hybrid CNN-GNN  |  Prototype GNN
trained with ArcFace / metric-learning losses
   │
   ▼  (5) Matching & evaluation
cosine similarity → Rank-k identification, EER, ROC AUC
```

**Key methodological choices**

- **Learned keypoints over SIFT.** Kornia-DISK descriptors replace handcrafted SIFT, giving more robust, illumination-tolerant nodes.
- **Bilinear feature sampling (the Hybrid model's novelty).** Instead of cropping image patches around each keypoint (expensive, and what the weaker `GNN++` variant does), the Hybrid model passes the whole image through a shared EfficientNet backbone once, then **samples the deep feature map at each keypoint coordinate via bilinear interpolation**. This gives every node a rich, contextual feature vector cheaply, which is then refined by Dynamic EdgeConv and a GATv2-based Topological Relation Module (TRM).
- **ArcFace loss.** Additive angular-margin loss (scale 128, margin 0.35) produces well-separated embeddings suited to the 260-class, high-intra-class-variation livestock setting.

## 5. Dataset

**Zenodo Beef Cattle Muzzle Database** (DOI: 10.5281/zenodo.6324361)

| Property | Value |
| :-- | :-- |
| Animals (classes) | 260 |
| Total images | 4,891 |
| Images per animal | 18.8 ± 10.0 (min 5, max 70, median 16) |
| Split (70/15/15) | 3,312 train / 615 val / 964 test |

The 964-image test set is the basis for all headline metrics.

## 6. Scope and Novel Contributions

1. **First controlled, like-for-like comparison** of CNN, pure GNN, Hybrid, and Prototype-GNN architectures on the same muzzle benchmark with the same protocol.
2. A **Hybrid CNN-GNN architecture** that fuses CNN texture (via bilinear feature-map sampling at keypoints) with learned graph topology.
3. **Deep learned keypoint graphs** (Kornia-DISK) replacing handcrafted keypoints.
4. **Dual explainability** (Grad-CAM + graph attention) to connect predictions back to muzzle anatomy.
5. A **statistically validated** result set (cross-validation + significance testing), not single-run numbers.

## 7. Technology Stack

- **ML / DL:** PyTorch, PyTorch-Geometric, Kornia (DISK), timm/EfficientNet.
- **Imaging:** OpenCV (CLAHE, Otsu), NumPy.
- **Evaluation & figures:** scikit-learn, Matplotlib (vector PDF + PNG output), LaTeX table generation.
- **Deployment (demonstrator):** FastAPI backend with PostgreSQL + `pgvector` similarity search, React frontend, Docker Compose.
- **Hardware:** single NVIDIA RTX 5070 (8 GB), mixed-precision (bfloat16) training.

## 8. Why This Matters

Beyond cattle, the framework — *learned keypoints → graph → GNN/Hybrid fusion → biometric matching* — generalises to any pattern-based biometric where topology matters (other livestock, wildlife re-identification, dermatoglyphics). The deployable web demonstrator shows the path from research result to a tool a farm or registry could actually use.

---

*Companion document: [PROGRESS_REPORT.md](PROGRESS_REPORT.md) — current status, results, and next steps.*
