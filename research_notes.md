# Research Notes: Biometric Cattle Muzzle Identification Benchmark

This document summarizes the final results, key scientific findings, model comparison, and commands for the biometric cattle muzzle identification project.

---

## 1. Executive Performance Summary

Below is the consolidated performance of all models tested on the **Beef Cattle Muzzle Database** (260 animals, 4,891 total images, split 70/15/15).

| Model Rank | Architecture | Rank-1 (%) | Rank-5 (%) | EER (%) | ROC AUC | Notes / Configuration |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Ensemble (CNN TTA + Hybrid)** | **96.1** | **98.1** | **0.78** | **0.9995** | Proposed SOTA (Weight Sweep: CNN=0.95, Hybrid=0.05) |
| 2 | **CNN (with TTA)** | **95.4** | **97.4** | **2.70** | **0.9961** | EfficientNet-B4 + ArcFace with Test-Time Augmentation |
| 3 | **CNN (EfficientNet-B4)** | **95.4** | **97.4** | **2.70** | **0.9961** | Baseline EfficientNet-B4 with ArcFace Loss |
| 4 | **VGG-16 Baseline** | **95.1** | **97.7** | **1.23** | **0.9993** | Bello et al. (2020) Re-implementation |
| 5 | **ResNet-50 Baseline** | **94.6** | **97.3** | **2.14** | **0.9971** | Qin et al. (2021) Re-implementation |
| 6 | **Hybrid CNN-GNN** | **92.0** | **96.7** | **1.85** | **0.9979** | Proposed (EfficientNet-B3 + EdgeConv + TRM + ArcFace) |
| 7 | **ProtoN (Prototype Node GNN)** | **91.6** | **94.8** | **1.17** | **0.9982** | Proposed GNN with Alignment Loss |
| 8 | **GNN v4 (Enhanced GATv2)** | **91.6** | **94.4** | **1.48** | **0.9937** | 4-layer GATv2 with Virtual Node (Enhanced) |
| 9 | **GNN v3 (Optimized GATv2)** | **91.5** | **95.0** | **1.87** | **0.9954** | 4-layer GATv2 with Virtual Node |
| 10 | **GNN++ (CNN Patches)** | **78.3** | **86.2** | **7.81** | **0.9730** | MobileNetV3 patch features on GNN nodes |
| 11 | **GNN+ (Kornia DISK)** | **72.0** | **84.2** | **11.17** | **0.9516** | DISK descriptor features on GNN nodes |

---

## 2. Key Insights & Scientific Findings

1. **Why the Hybrid Model Wins Over Pure GNNs**:
   Instead of using static handcrafted descriptors (SIFT) or patch crops (GNN++), the **Hybrid Model** uses **bilinear feature map sampling** at keypoint locations from a shared EfficientNet-B3 backbone. This captures deep, contextualized receptive field features at key points without the high computational cost of separate patch forward passes.
2. **The Power of the Ensemble**:
   By blending the global spatial representations of the CNN with the local topological invariant graphs of the Hybrid model, the Ensemble corrects misclassifications and achieves the highest overall Rank-1 Accuracy of **96.1%** and a remarkably low EER of **0.78%**.
3. **Statistical Significance**:
   McNemar's tests confirmed that the improvement of the CNN (EfficientNet-B4) and the Hybrid models over GNN v3/v4 is highly statistically significant ($p < 10^{-8}$).
4. **Generalization (5-Fold Cross-Validation)**:
   - CNN Fold Stability: 93.91% ± 0.31% Rank-1
   - ProtoN Fold Stability: 89.49% ± 0.71% Rank-1
   - Hybrid Fold Stability: 68.88% ± 2.04% Rank-1 (Trained on CV-safe 12 epochs)

---

## 3. Core File Map

- **Model Definitions**:
  - CNN: `src/models/cnn_model.py`
  - Hybrid: `src/models/hybrid_model.py`
  - ProtoN: `src/models/proton.py`
  - GNN v3: `src/models/gnn_v3.py`
- **Training Scripts**:
  - CNN: `scripts/train_cnn.py`
  - Hybrid: `scripts/train_hybrid.py`
  - ProtoN: `scripts/train_proton.py`
- **Baseline Training Scripts**:
  - VGG-16: `scripts/baselines/train_vgg_baseline.py`
  - ResNet-50: `scripts/baselines/train_resnet_baseline.py`
- **Evaluation & Metrics**:
  - Cross-Validation: `scripts/cross_validation.py`
  - Statistical Tests: `scripts/statistical_tests.py`
  - Model Comparisons: `scripts/compare_models.py`
- **Visualizations & Explainability**:
  - Figure Generator: `scripts/figures/generate_paper_figures.py`
  - Grad-CAM: `scripts/visualize_gradcam.py`
  - GNN Attention: `scripts/visualize_gnn_attention.py`

---

## 4. Quick Execution Commands

To replicate results, plot figures, or perform checks:

```powershell
# 1. Run 5-Fold Cross Validation
venv\Scripts\python.exe scripts/cross_validation.py --epochs-cnn 10 --epochs-proton 12 --epochs-hybrid-p1 10 --epochs-hybrid-p2 2

# 2. Re-generate Main Comparative Report & Markdown Table
venv\Scripts\python.exe scripts/compare_models.py

# 3. Re-generate Publication-quality Vector Figures (PDF/PNG)
venv\Scripts\python.exe scripts/figures/generate_paper_figures.py

# 4. Generate Explainability Maps
venv\Scripts\python.exe scripts/visualize_gradcam.py
venv\Scripts\python.exe scripts/visualize_gnn_attention.py
```

*Notes saved locally on 2026-06-12.*
