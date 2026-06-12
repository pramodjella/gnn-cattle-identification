# 🐄 Biometric Cattle Identification via Deep Learning & Graph Neural Networks

A complete pipeline for **individual cattle identification** using muzzle print patterns, combining **learned keypoints (Kornia-DISK)**, **Graph Neural Networks (GATv2 + EdgeConv)**, **EfficientNet CNNs**, and **bilinear feature map sampling**.

This project provides a comprehensive comparative benchmark of pure CNN, Graph Neural Network (GNN), Hybrid CNN-GNN, and Prototype GNN architectures on the Zenodo Beef Cattle Muzzle Database (260 animals, 4,891 images). It generates publication-quality figures (ROC, CMC, t-SNE, explainability maps) and LaTeX tables suitable for precision agriculture journals such as *Computers and Electronics in Agriculture*.

---

## 🚀 Performance Benchmark

Below is the consolidated performance on the test split (964 images) across all implemented proposed architectures and literature baselines:

| Model | Rank-1 (%) | Rank-5 (%) | EER (%) | ROC AUC | Description / Reference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ensemble (CNN TTA + Hybrid)** | **96.1** | **98.1** | **0.78** | **0.9995** | Proposed SOTA (Blended weights: CNN=0.95, Hybrid=0.05) |
| CNN (with TTA) | 95.4 | 97.4 | 2.70 | 0.9961 | EfficientNet-B4 + ArcFace with Test-Time Augmentation |
| CNN (EfficientNet-B4) | 95.4 | 97.4 | 2.70 | 0.9961 | Baseline EfficientNet-B4 with ArcFace Loss |
| VGG-16 Baseline | 95.1 | 97.7 | 1.23 | 0.9993 | Re-implementation of Bello et al. (2020) |
| ResNet-50 Baseline | 94.6 | 97.3 | 2.14 | 0.9971 | Re-implementation of Qin et al. (2021) |
| Hybrid CNN-GNN | 92.0 | 96.7 | 1.85 | 0.9979 | Proposed (Bilinear Feature Sampling + EdgeConv + TRM) |
| ProtoN (Prototype Node GNN) | 91.6 | 94.8 | 1.17 | 0.9982 | Proposed GNN with Cross-Graph Alignment Loss |
| GNN v4 (GATv2 - Enhanced) | 91.6 | 94.4 | 1.48 | 0.9937 | 4-layer GATv2 with Virtual Node (Enhanced) |
| GNN v3 (GATv2 + VN) | 91.5 | 95.0 | 1.87 | 0.9954 | 4-layer GATv2 with Virtual Node |
| GNN++ (CNN Patches) | 78.3 | 86.2 | 7.81 | 0.9730 | MobileNetV3 patch features on GNN nodes |
| GNN+ (Kornia DISK) | 72.0 | 84.2 | 11.17 | 0.9516 | DISK descriptor features on GNN nodes |

---

## 📁 Project Structure

```
gnn-cattle-identification/
├── config/
│   └── config.yaml              # Hyperparameters (tuned settings)
├── data/
│   ├── raw/                     # Original dataset (downloaded)
│   └── preprocessed/            # After CLAHE, segmentation
├── src/
│   ├── preprocessing/           # ROI, CLAHE, segmentation
│   ├── features/                # Learned DISK, KNN graph builder
│   ├── models/                  # EdgeConv, TRM, CattleGNN, losses
│   ├── training/                # Dataset loaders, trainer, augmentation
│   └── evaluation/              # Metrics, visualization
├── scripts/
│   ├── baselines/
│   │   ├── train_vgg_baseline.py    # Bello et al. (2020) VGG-16 baseline
│   │   └── train_resnet_baseline.py # Qin et al. (2021) ResNet-50 baseline
│   ├── figures/
│   │   └── generate_paper_figures.py # Main figures generator (vector PDF + PNG)
│   ├── 01_download_data.py      # Download Zenodo dataset
│   ├── 02_preprocess.py         # ROI + CLAHE + segmentation
│   ├── 03_extract_keypoints.py  # Learned keypoints extraction
│   ├── 04_build_graphs.py       # KNN graph construction
│   ├── train_cnn.py             # EfficientNet-B4 + ArcFace training
│   ├── train_hybrid.py          # Feature map sampled Hybrid GNN training
│   ├── train_proton.py          # ProtoN GNN training
│   ├── cross_validation.py      # Stratified 5-Fold cross-validation loops
│   ├── compare_models.py        # Compiles comparison tables & main curves
│   ├── statistical_tests.py     # Pairwise McNemar tests & Bootstrap CIs
│   ├── visualize_gradcam.py     # CNN Grad-CAM explainability generator
│   └── visualize_gnn_attention.py # TRM Attention heatmap visualizer
├── outputs/
│   ├── figures/                 # Paper-ready plots (CMC, ROC, training, t-SNE)
│   ├── stats/                   # JSON statistics per model & ablation results
│   └── results/                 # Publication MD report
├── paper_draft.md               # Full manuscript draft for journal submission
├── research_notes.md            # Quick reference local notes file
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (highly recommended; fallbacks to CPU)

### Setup
```bash
# Clone the repository
git clone https://github.com/pramodjella/gnn-cattle-identification.git
cd gnn-cattle-identification

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.txt
```

---

## 🛠️ Usage Workflow

### 1. Data Preparation
```bash
# Download dataset (4,891 images, 260 cattle)
python scripts/01_download_data.py

# Preprocess (ROI extraction + CLAHE + Otsu segmentation)
python scripts/02_preprocess.py

# Extract learned DISK keypoints & descriptors
python scripts/03_extract_keypoints.py

# Build KNN graphs (k=8)
python scripts/04_build_graphs.py
```

### 2. Tuned Model Training
```bash
# Train EfficientNet-B4 CNN
python scripts/train_cnn.py

# Train ProtoN GNN
python scripts/train_proton.py

# Train Hybrid CNN-GNN (bilinear sampling + EdgeConv + TRM)
python scripts/train_hybrid.py
```

### 3. Prior Art Baseline Training
```bash
# Train VGG-16 Baseline (Bello et al. 2020)
python scripts/baselines/train_vgg_baseline.py

# Train ResNet-50 Baseline (Qin et al. 2021)
python scripts/baselines/train_resnet_baseline.py
```

### 4. Cross-Validation & Statistical Tests
```bash
# Run Stratified 5-Fold Cross-Validation on top 3 models
python scripts/cross_validation.py

# Run McNemar's significance tests and Bootstrap Confidence Intervals
python scripts/statistical_tests.py
```

### 5. Generate Figures & LaTeX Tables
```bash
# Generate model comparison report
python scripts/compare_models.py

# Generate publication-quality figures (CMC, ROC, training, t-SNE)
python scripts/figures/generate_paper_figures.py

# Generate explainability maps (Grad-CAM and GNN Attention)
python scripts/visualize_gradcam.py
python scripts/visualize_gnn_attention.py
```

---

## 📊 Visualizations

All generated visuals are saved under `outputs/figures/`:
- **CMC & ROC Curves**: Evaluates identification and verification rates.
- **t-SNE Embeddings**: Clusters represent individual cattle in the learned feature spaces.
- **Dual Explainability**: Visualizes Grad-CAM activations on raw images side-by-side with TRM GNN attention weights.

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{muzzle-biometrics-gnn,
  title={Graph-Augmented Deep Learning for Cattle Muzzle Biometric Identification: A Comparative Study of CNN, Hybrid, and Graph Neural Network Architectures},
  author={Jella, Pramod},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  note={DISK keypoints + Dynamic EdgeConv + TRM + ArcFace loss pipeline}
}
```
