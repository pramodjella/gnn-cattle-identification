# 🐄 Biometric Cattle Identification via Deep Learning & Graph Neural Networks

A complete pipeline for **individual cattle identification** using muzzle print patterns, combining **SuperPoint keypoint detection**, **KNN graph construction**, **EdgeConv-based Graph Neural Networks**, and **LightGlue-inspired matching**.

> **Paper-ready**: Generates publication-quality figures (ROC, CMC, t-SNE, score distributions) and LaTeX tables for international conferences (WACV, ICPR, ICIP, Computers & Electronics in Agriculture).

---

## Architecture

```
Raw Muzzle Image
       │
       ▼
┌──────────────────┐
│  Preprocessing   │  ROI Extraction → CLAHE → Otsu Segmentation
└────────┬─────────┘
         ▼
┌──────────────────┐
│    SuperPoint     │  Keypoint Detection + 256-d Descriptors
└────────┬─────────┘
         ▼
┌──────────────────┐
│  KNN Graph (k=12)│  Nodes = Keypoints, Edges = Spatial Neighbors
└────────┬─────────┘
         ▼
┌──────────────────┐
│    CattleGNN     │  EdgeConv × 3 → TRM (GAT) → Mean+Max Pool → 256-d Embedding
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Matching      │  Cosine Similarity + Hungarian + Threshold Calibration
└──────────────────┘
```

## Project Structure

```
gnn-cattle-identification/
├── config/
│   └── config.yaml              # All hyperparameters
├── data/
│   ├── raw/                     # Original dataset (downloaded)
│   ├── preprocessed/            # After CLAHE, segmentation
│   └── graphs/                  # Serialized PyG graph data
├── src/
│   ├── preprocessing/           # ROI, CLAHE, segmentation
│   ├── features/                # SuperPoint, KNN graph builder
│   ├── models/                  # EdgeConv, TRM, CattleGNN, losses
│   ├── training/                # Dataset, trainer, triplet mining
│   ├── matching/                # LightGlue matcher, verification
│   ├── evaluation/              # Metrics, visualization
│   └── utils.py                 # Config, logging, reproducibility
├── scripts/
│   ├── 01_download_data.py      # Download Zenodo dataset
│   ├── 02_preprocess.py         # ROI + CLAHE + segmentation
│   ├── 03_extract_keypoints.py  # SuperPoint extraction
│   ├── 04_build_graphs.py       # KNN graph construction
│   ├── 05_train.py              # GNN training (triplet + CE loss)
│   ├── 06_evaluate.py           # Full biometric evaluation
│   └── 07_generate_paper_stats.py  # LaTeX tables & statistics
├── outputs/
│   ├── checkpoints/             # Model weights
│   ├── logs/                    # Training logs
│   ├── figures/                 # Paper-quality plots
│   └── stats/                   # JSON statistics per phase
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended; CPU fallback supported)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/gnn-cattle-identification.git
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

> **Note**: Install PyTorch and PyG first with CUDA support matching your GPU drivers. See [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/install/) installation guides.

## Usage

Run the pipeline scripts sequentially:

```bash
# Phase 1: Download dataset (4923 images, 268 cattle)
python scripts/01_download_data.py

# Phase 2: Preprocess (ROI + CLAHE + Segmentation)
python scripts/02_preprocess.py

# Phase 3: Extract SuperPoint keypoints & descriptors
python scripts/03_extract_keypoints.py

# Phase 4: Build KNN graphs
python scripts/04_build_graphs.py

# Phase 5-6: Train CattleGNN (triplet loss + CrossEntropy)
python scripts/05_train.py

# Phase 7-8: Evaluate & generate paper figures
python scripts/06_evaluate.py
python scripts/07_generate_paper_stats.py
```

## Dataset

**Beef Cattle Muzzle Database** from Zenodo  
- **DOI**: [10.5281/zenodo.6324361](https://zenodo.org/records/6324361)  
- **Size**: 4,923 muzzle images from 268 beef cattle  
- **Split**: 70% train / 15% val / 15% test (stratified by animal)

The download script will attempt to fetch the dataset automatically. If it fails, manually download the zip from Zenodo and place it in `data/raw/`.

## Model Architecture

### CattleGNN

| Component | Details |
|-----------|---------|
| **Input** | SuperPoint descriptors (256-d) per keypoint |
| **EdgeConv** | 3 layers of Dynamic EdgeConv with residual connections |
| **TRM** | Multi-head Graph Attention (4 heads, 2 layers) with GraphNorm |
| **Pooling** | Global Mean + Max pooling → 512-d |
| **Projection** | FC → 256-d L2-normalized embedding |
| **Classification** | Optional FC head for auxiliary CrossEntropy loss |

### Training

| Parameter | Value |
|-----------|-------|
| Loss | Triplet (margin=0.5) + CE (weight=0.5) |
| Mining | Online hard negative mining |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine annealing with warm restarts |
| Augmentation | Random keypoint dropout (10%) |
| Early Stopping | Patience=15, min_delta=0.001 |

## Evaluation Metrics

The evaluation pipeline computes:

- **Identification**: Rank-1, Rank-5, Rank-10 accuracy, CMC curves
- **Verification**: TAR @ FAR={0.1%, 1%, 10%}, EER, ROC AUC
- **Score Analysis**: Genuine/impostor distributions, d-prime separability
- **Embeddings**: t-SNE visualization of learned representations

## Configuration

All hyperparameters are in `config/config.yaml`. Key settings:

```yaml
model:
  edge_conv:
    num_layers: 3
    hidden_dims: [256, 256, 512]
    k_dynamic: 12
  trm:
    num_heads: 4
    num_layers: 2
  embedding_dim: 256

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  triplet:
    margin: 0.5
    mining_type: hard
```

## Citation

If you use this work, please cite:

```bibtex
@misc{cattle-gnn-biometric,
  title={Biometric Cattle Identification via Graph Neural Networks on Muzzle Print Patterns},
  year={2026},
  note={SuperPoint keypoints + EdgeConv GNN + LightGlue matching pipeline},
}
```

## License

This project is for academic research purposes.
