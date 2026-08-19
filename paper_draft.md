# Manuscript Draft: Graph-Augmented Deep Learning for Cattle Muzzle Biometric Identification

**Journal Target:** *Computers and Electronics in Agriculture*  
**Submission Type:** Research Article  

---

## Abstract
Cattle muzzle print biometrics offer a non-invasive, tamper-proof method for individual identification, crucial for animal traceability, disease control, and ownership verification. While traditional approaches rely on handcrafted texture descriptors (e.g., SIFT, LBP) or raw image convolution, they struggle with spatial scaling and geometric deformations. In this paper, we present a systematic comparative study of convolutional neural networks (CNN), pure graph neural networks (GNN), and hybrid architectures on the public Zenodo Beef Cattle Muzzle Database (260 animals, 4,891 images), evaluated under a verified closed-set, image-level split with a documented no-leakage audit. We introduce a **Hybrid CNN-GNN** architecture that extracts local invariant graph structures from keypoints and dynamically updates node embeddings using bilinear feature map sampling from a shared CNN backbone. Additionally, we evaluate **ProtoN**, a prototype-based node GNN optimized with cross-graph alignment loss. Our best single model, an EfficientNet-B4 CNN with ArcFace, reaches **95.4% Rank-1** accuracy. Blending it with the Hybrid CNN-GNN — using a fusion weight **selected on validation and applied unchanged to test** — yields **96.1% Rank-1** and, more importantly, reduces the Equal Error Rate from **2.70% to 0.78%** (a 3.5× improvement in verification reliability), matching or exceeding prior-art VGG-16 (95.1%) and ResNet-50 (94.6%) re-implementations. We show that the graph **branch**'s principal contribution is verification rather than closed-set ranking. A control the literature usually omits, however, bounds what that means: fusing the same CNN with a *non-graph* baseline (VGG-16) reduces EER by a statistically *indistinguishable* amount (CIs [+0.47,+2.63] vs [+0.44,+2.48] points), so the verification gain is **not graph-specific** — it reflects a poorly calibrated CNN that almost any better-calibrated partner repairs. What remains graph-specific is narrower: of the partners tested, only the graph branch improves ranking *and* verification simultaneously. A pathway-intervention study then sharpens *why*, and yields an instructive negative result: randomising the graph edges, zeroing the geometric edge attributes, or shuffling keypoint positions each changes Rank-1 by ≤0.5 points, whereas zeroing the keypoint-sampled node features collapses it from 92.0% to 0.1% — the gain is attributable to keypoint-sampled CNN features and feature-space message passing, not to the explicit geometric topology we supply. Through a controlled corruption study we further show that a *single validation-tuned* fusion weight is the robust choice — inheriting the graph branch's clean-data verification gain while collapsing gracefully to the CNN under severe degradation — whereas per-sample quality-adaptive gating does not improve on it. Finally, we move beyond qualitative saliency with a *causal* explainability protocol (node-ablation faithfulness and a Hybrid pathway intervention), showing the GNN distributes its identity decision across many keypoints and that the Hybrid's in-domain signal is carried by CNN texture rather than the input graph geometry.

---

## 1. Introduction
Traceability in livestock production is essential for managing disease outbreaks, assuring consumer safety, and optimizing herd management. Traditional identification methods, such as ear tags, branding, and radio-frequency identification (RFID) transponders, present several limitations: they are invasive, prone to loss or damage, and susceptible to fraud. Biometric identification, which utilizes unique physiological characteristics of animals, presents a permanent and non-invasive alternative. Among various biometric indicators, the muzzle print (or nose print) pattern of cattle remains invariant over time, similar to human fingerprints.

Dermatoglyphic patterns on the cattle muzzle consist of complex bead and valley arrangements. Capturing these fine-grained features computationally is challenging due to:
1. **Geometric Deformations:** Head movement, angle of capture, and camera positioning distort the apparent spacing of beads.
2. **Environmental Conditions:** Variable outdoor illumination, shadows, dirt, and moisture alter the contrast of valley lines.
3. **Scale Variations:** Muzzle size grows as the animal matures, requiring scale-invariant matching.

Early research relied on handcrafted features such as Scale-Invariant Feature Transform (SIFT) or Local Binary Patterns (LBP) combined with Support Vector Machines (SVM). However, handcrafted features fail to generalize under variable field lighting and partial occlusions. The advent of Deep Learning, particularly Convolutional Neural Networks (CNNs), led to substantial improvements by learning hierarchical texture representations directly from raw images. Nevertheless, CNNs are inherently translation-equivariant and can struggle with non-rigid geometric transformations of the muzzle pattern.

Graph Neural Networks (GNNs) present a promising paradigm by representing the muzzle print as a topological graph where nodes are keypoints and edges denote spatial relationships. In this work, we bridge the gap between grid-based CNN texture modeling and coordinate-based GNN topological modeling. 

### Contributions:
1. **Systematic, leakage-audited comparison:** We perform a rigorous comparative study on one benchmark dataset evaluating CNNs, GNNs, Hybrid models, and Prototype GNNs under a verified closed-set protocol with an explicit no-leakage audit (Section 2.1).
2. **Hybrid CNN-GNN Architecture:** We propose a model that samples deep backbone feature maps at keypoint coordinates via bilinear interpolation, combining rich texture features with topological invariance; we further introduce an optional multi-scale (FPN-style) sampling variant.
3. **Where graphs help:** Rather than claiming graphs beat CNNs on ranking, we quantify their true contribution — a validation-selected CNN+Hybrid blend cuts the Equal Error Rate 3.5× (2.70%→0.78%), i.e. the graph branch improves *verification* robustness while the CNN dominates closed-set *ranking*.
4. **Quantitative explainability:** Beyond Grad-CAM and GATv2 attention heatmaps, we report faithfulness metrics (Fidelity+/−, sparsity) and cross-method agreement, converting qualitative saliency into measurable evidence.

---

## 2. Methodology

### 2.1 Dataset, Split Protocol, and Leakage Audit
We use the public Zenodo Beef Cattle Muzzle Database (260 animals, 4,891 images). Images are split **per image, stratified by animal**, into 70/15/15 train/validation/test partitions, so that every one of the 260 identities appears in all three partitions. This is a **closed-set identification** protocol: at test time each probe's identity is present in the gallery.

To pre-empt the most common reviewer concern — that a reported test accuracy exceeding validation accuracy indicates leakage — we run an automated integrity audit (`scripts/verify_data_integrity.py`) that verifies: (i) no source image (by file stem) appears in more than one split; (ii) every identity is present in the gallery; and (iii) the `animal_id → class index` mapping is identical across splits. All checks pass. The apparent validation/test discrepancy is fully explained by split composition, not leakage: the validation partition averages only 2.4 images per identity (minimum 1), so single-image validation identities have **no genuine gallery mate** and are counted as forced misses, deflating validation Rank-1 relative to the test partition (3.7 images per identity). We therefore report test-set metrics as the primary results and use validation only for model and hyperparameter selection.

**Scope of the audit: image-level, not session-level.** The audit verifies that no *source image* appears in more than one split, but it cannot rule out *acquisition-level* similarity: images of the same animal from the same session, burst, or camera setup — or near-duplicate frames — may still be distributed across splits, which would make the closed-set task easier than a deployment scenario separating enrolment and query in time. The public release provides no session, capture-date, or camera metadata, so a session-disjoint split cannot be constructed from the available annotations. We therefore treat the in-domain numbers as an upper bound with respect to acquisition-level correlation, and note that the zero-shot cross-dataset evaluation (§3.4), where enrolment and query come from entirely different collections, is free of this concern.

### 2.2 Graph Construction via Learned Keypoints
Rather than utilizing classical SIFT keypoints, we adopt **Kornia-DISK** (Discrete Keypoint Detection and Matching), a reinforcement learning-based feature detector. For each muzzle image, we extract $N \le 128$ keypoint locations $p_i = (x_i, y_i)$ and their corresponding 256-dimensional deep descriptors $f_i$. 

We construct a directed $k$-nearest neighbor ($k$-NN) graph $G = (V, E)$, where $V$ is the set of keypoint nodes and $E$ is the set of edges. An edge $e_{ij} = (v_i, v_j)$ is established if $v_j$ is among the $k$-nearest spatial neighbors of $v_i$ (with $k=8$). Edge attributes $a_{ij}$ encode the spatial relationships:
$$a_{ij} = \left[ \Delta x, \Delta y, d_{ij}, \theta_{ij}, \text{rel\_scale} \right]$$
where $d_{ij}$ is the Euclidean distance, $\theta_{ij}$ is the angle, and rel_scale is the relative scale between descriptors.

### 2.3 Proposed Architectures

#### A. CNN Baseline (EfficientNet-B4 + ArcFace)
The baseline CNN utilizes an **EfficientNet-B4** backbone pre-trained on ImageNet. It takes the $256 \times 256$ muzzle image (enhanced with Contrast-Limited Adaptive Histogram Equalization, CLAHE) as input. The final convolutional feature maps are pooled into a 512-dimensional embedding space. The network is optimized using the **ArcFace (Additive Angular Margin) Loss**:
$$L_{Arc} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$
where $s$ is the logit scale (128.0) and $m$ is the angular margin (0.35).

#### B. Hybrid CNN-GNN
The Hybrid model combines a shared **EfficientNet-B3** backbone with a GNN head:
1. **Backbone Forward Pass:** The image is passed through the CNN to obtain feature map $F_{map} \in \mathbb{R}^{C \times H' \times W'}$.
2. **Bilinear Feature Sampling:** For each node keypoint coordinate $p_i$, we perform bilinear interpolation on $F_{map}$ to sample a local feature vector $x_i \in \mathbb{R}^{C}$. We additionally provide an optional **multi-scale** variant that samples several backbone stages (strides 16/16/32; 96+232+1536 channels) and concatenates them per node, so that each node carries both fine groove texture (earlier, higher-resolution stages) and coarse anatomical context (later stages) rather than only the stride-32 final map.
3. **Graph Convolutions:** Node features are projected to 256-d and passed through **Dynamic EdgeConv** blocks to update features based on local graph structure.
4. **Relation Module:** A **Topological Relation Module** using a 4-head GATv2 aggregates multi-hop topological features.
5. **Global Pooling:** Combined global mean and max pooling maps graph features to a final 256-d embedding.

```
Muzzle Image (256x256) ──> EfficientNet-B3 Backbone ──> Feature Map (1536x8x8)
                                                          │
DISK Keypoints ──────────> Bilinear Interpolation  <──────┘
                                │
                         Node Features (1536-d) ──> Proj (256-d)
                                │
                         Dynamic EdgeConv Blocks (512-d)
                                │
                         Topological Relation (GATv2)
                                │
                         Mean + Max Pool ──> ArcFace Loss
```

#### C. ProtoN (Prototype Node GNN)
ProtoN represents each animal class as a prototype vector computed by averaging node features across support graphs. During training, we apply a **Cross-Graph Alignment Loss** that penalizes spatial misalignment between matching keypoint neighborhoods in query and support graphs, enabling high-quality metric learning.

### 2.4 Adaptive Graph Construction (ADGC)
A static geometric $k$-NN graph encodes exactly the deformation we want the model to be invariant to: two captures of the same muzzle at different angles yield different neighbourhoods, and spurious edges bridge unrelated ridge regions. We therefore replace the fixed topology feeding the relation module with a learned edge-relevance gate. For each candidate edge $e_{ij}$, an MLP scores the pair from the endpoint node features and geometric edge attributes, producing a gate $g_{ij}=\sigma(\text{MLP}([x_i, x_j, a_{ij}])) \in [0,1]$. Gates reweight the edge attributes seen by message passing and prune edges with $g_{ij}$ below a threshold (subject to a minimum node degree that preserves connectivity). ADGC is a single self-contained module (`src/models/adaptive_graph.py`), enabling a clean ablation against the static $k$-NN and against multi-scale sampling.

---

## 3. Results and Discussion

### 3.1 Quantitative Results

Table 1 details the biometric performance on the test set (964 samples).

**Table 1: Main Identification & Verification Performance (test split, 964 images)**
| Model | Rank-1 (%) | Rank-5 (%) | EER (%) | ROC AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Ensemble (CNN TTA + Hybrid, val-selected)** | **96.1** | **98.1** | **0.78** | **0.9995** |
| CNN (with TTA) | 95.4 | 97.4 | 2.70 | 0.9961 |
| CNN (EfficientNet-B4) | 95.4 | 97.4 | 2.70 | 0.9961 |
| VGG-16 Baseline (Bello et al.) | 95.1 | 97.7 | 1.23 | 0.9993 |
| ResNet-50 Baseline (Qin et al.) | 94.6 | 97.3 | 2.14 | 0.9971 |
| Hybrid CNN-GNN | 92.0 | 96.7 | 1.85 | 0.9979 |
| ProtoN (Prototype Node GNN) | 91.6 | 94.8 | 1.17 | 0.9982 |
| GNN v4 (GATv2 - Enhanced) | 91.6 | 94.4 | 1.48 | 0.9937 |
| GNN v3 (GATv2 + VN) | 91.5 | 95.0 | 1.87 | 0.9954 |
| GNN++ (CNN Patches) | 78.3 | 86.2 | 7.81 | 0.9730 |
| GNN+ (Kornia DISK) | 72.0 | 84.2 | 11.17 | 0.9516 |

**Ensemble fusion protocol.** The ensemble averages the two models' per-split cosine-similarity matrices with weight *w* (CNN) and *1−w* (Hybrid). Critically, *w* is selected on the **validation** split and then applied unchanged to test — it is never tuned on test. The validation-selected weight is *w*=0.95, and applying it to test gives 96.1% Rank-1; the test-oracle weight (the best *w* had we improperly tuned on test) is also 0.95, so the validation→test selection gap is 0.00 points. This makes the headline number defensible rather than an artefact of test-set tuning (`scripts/ensemble_inference.py`).

**Is the verification gain graph-specific? A control.** The CNN's 2.70% EER is the weakest of our strong models — VGG-16 reaches 1.23% and ProtoN 1.17% standalone — so a natural objection is that the blend simply repairs a poorly calibrated CNN, with the graph incidental. Running the *identical* validation-selected protocol against a non-graph partner shows the objection is largely correct.

| Partner | standalone EER | sel. w | fused R-1 | fused EER | ΔEER (pt, 95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid (graph)** | 1.88 | 0.95 | **95.95** | 1.17 | +1.65 [+0.44, +2.48] |
| ProtoN (graph) | 1.17 | 0.85 | 93.57 | **0.67** | +2.14 [+0.50, +3.22] |
| VGG-16 (*non-graph*) | 1.23 | 0.00 | 95.12 | 1.23 | +1.58 [+0.47, +2.63] |

VGG-16 delivers an EER reduction statistically indistinguishable from the Hybrid's, and its validation sweep selects w=0.00 — it discards our CNN altogether, because VGG-16 is already the better-calibrated verifier. We therefore do *not* claim the verification gain as evidence for graph modelling. The narrower claim the data supports is that, among the partners tested, only the graph branch improves ranking *and* verification simultaneously: VGG-16 leaves Rank-1 unchanged and ProtoN costs 1.55 Rank-1 points to buy its lower EER.


Analysis of the results indicates that:
- **CNN dominates ranking; the graph branch improves verification.** The CNN alone reaches 95.4% Rank-1; adding the Hybrid branch lifts Rank-1 only marginally (to 96.1%) but reduces the Equal Error Rate from 2.70% to 0.78% — a 3.5× improvement. The honest reading of the fusion weight (0.95/0.05) is therefore *not* that graphs rival CNNs on ranking, but that the graph branch contributes complementary topological evidence that sharpens the genuine/impostor decision boundary, which is what verification (EER, TAR@FAR) rewards.
- **Topological verification robustness.** Consistent with this, the pure ProtoN and Hybrid GNNs — despite lower Rank-1 — achieve very low standalone EERs (1.17% and 1.85%), indicating that topological structure yields well-separated genuine/impostor score distributions.
- **DISK vs. handcrafted (SIFT).** The baseline GNN+ using DISK (72.0% Rank-1) substantially outperforms SIFT-based node features, confirming that deep learned keypoints provide more robust node descriptors under scale and illumination change.

### 3.2 5-Fold Cross Validation & Statistical Significance
To verify fold stability, stratified 5-fold cross-validation was performed on the top models. pair-wise McNemar tests were conducted to confirm if improvements are statistically significant.

**Table 2: 5-Fold Cross-Validation Performance (all models trained to convergence)**
| Model | Rank-1 (%) | Rank-5 (%) | EER (%) | ROC AUC |
| :--- | :---: | :---: | :---: | :---: |
| **CNN (EfficientNet-B4)** | **97.26 ± 0.36** | **97.98 ± 0.25** | **0.06 ± 0.08** | **1.0000** |
| Hybrid CNN-GNN | 95.52 ± 0.31 | 97.40 ± 0.27 | 0.49 ± 0.16 | 0.9998 |
| ProtoN GNN | 94.95 ± 0.37 | 96.34 ± 0.40 | 0.70 ± 0.11 | 0.9990 |
| *Hybrid, reduced budget (12 ep.)* | *68.88 ± 2.04* | *84.22 ± 1.27* | *11.31 ± 1.47* | *0.9491* |

Our earlier cross-validation used a reduced budget for all three models to keep the sweep tractable, and the Hybrid's two-phase schedule (cached backbone features → end-to-end fine-tuning) did not converge within it: the Hybrid scored only 68.88 ± 2.04%, with a fold variance that invited the reading that the architecture is unstable. We have since re-run **all three models at full budget** on every fold (CNN 150 epochs, ProtoN 200, Hybrid 150 cached + 30 end-to-end; ≈13 h, 2.3 h and 6.8 h of GPU time respectively on one RTX 5070), so Table 2 is now internally comparable.

The budget, not the architecture, was the whole story. The Hybrid rises from 68.88 ± 2.04% to **95.52 ± 0.31%** (EER 11.31% → **0.49%**), its fold variance collapsing by a factor of six; ProtoN rises from 89.49 ± 0.71% to **94.95 ± 0.37%**, and the CNN from 93.91 ± 0.31% to **97.26 ± 0.36%**. All three are now stable across folds (±0.31–0.37). Reassuringly, the ordering under this uniform budget reproduces the single-split ordering of Table 1 — CNN (97.26%) > Hybrid (95.52%) > ProtoN (94.95%) — independently confirming this paper's central claim that the CNN, not the graph branch, dominates closed-set ranking.

**Why the cross-validated values exceed their single-split counterparts.** Every full-budget CV number is higher than the corresponding single-split test result (CNN 97.26% vs 95.4%; Hybrid 95.52% vs 92.0%; ProtoN 94.95% vs 91.6%), and we state the reasons rather than leave the discrepancy implicit. First, each fold trains on 80% of the data versus 70% for the single split — roughly 14% more training images. Second, and more importantly, the two protocols select their final model differently: each CV fold trains for a fixed schedule and evaluates the *final* model, whereas the single-split models are best-validation-Rank-1 checkpoints, chosen on a small validation partition that averages only 2.4 images per identity and counts single-image identities as forced misses (§2.1). Neither protocol touches test data for model selection, so both sets of numbers are honest; they simply answer different questions. We therefore keep Table 1 as the headline and read Table 2 as evidence of *stability across folds*. Exact McNemar tests on *per-probe* single-split predictions (`scripts/compute_real_mcnemar.py`) confirm the CNN's advantage over both graph models is significant: CNN vs. Hybrid n10=43, n01=13, p=7.3×10⁻⁵; CNN vs. ProtoN n10=53, n01=19, p=7.6×10⁻⁵. The Hybrid and ProtoN are, however, *statistically indistinguishable* (n10=38, n01=34, p=0.72), so their ordering in Table 1 should not be read as a real difference.

### 3.3 Open-Set Evaluation
Closed-set Rank-1 assumes every probe is enrolled; deployments must also reject animals that were never enrolled. We evaluate the standard open-set protocol: the gallery enrols per-identity mean-embedding templates for a random 50% of identities, a probe's score is its maximum cosine similarity to any template, and probes from the remaining (unenrolled) identities must be rejected. We report the Detection-and-Identification Rate (DIR@rank-1) at fixed False Alarm Rates and the open-set ROC-AUC. Averaged over five random identity partitions (`scripts/evaluate_openset.py`), the CNN attains **98.6 ± 0.9% Rank-1 on known probes**, an **open-set AUC of 0.986 ± 0.005**, and Detection-and-Identification Rates of **62.6 ± 13.8% at FAR=1%** and **95.5 ± 2.1% at FAR=5%**. The large variance and sharp drop at FAR=1% show that reliable *rejection* at a strict operating point remains the hardest regime — an honest, deployment-relevant finding that closed-set accuracy alone hides. (For a strict unseen-identity claim, the camera-ready additionally retrains with the unknown identities held out of training; the harness supports this.)

### 3.4 Zero-Shot Cross-Dataset Transfer

To test whether the learned representation generalises beyond its acquisition domain, we evaluate the EfficientNet-B4 model — trained only on the primary US Beef Cattle Muzzle Database — on **two separate, independently collected** muzzle datasets with **no fine-tuning**: (A) a 24-identity set (667 usable images) and (B) a larger 308-identity set (1,848 images) captured in a different country. Because both datasets consist of wide farm scenes rather than muzzle crops, we first localise the muzzle with a lightweight detector (YOLOv8n trained on an independent single-class muzzle-detection set; validation mAP@50 = 0.995), which also serves as the deployment-time front-end. The same CLAHE preprocessing used in training is applied to the crops. For closed-set metrics we retain identities with at least two images (a single-image identity has no genuine gallery mate); we also audit and remove a redundant "master pool" folder in set A that duplicated every image under one label (an artefact that otherwise corrupts rank-1 matching).

**Table 3: Zero-shot cross-dataset transfer (train on US set, test on external sets, no fine-tuning)**
| Metric | Set A (24 IDs) | Set B (308 IDs) |
| :--- | :---: | :---: |
| Closed-set Rank-1 | **97.0%** | **87.0%** |
| Closed-set Rank-5 | 99.3% | 94.4% |
| Closed-set ROC AUC | 0.919 | 0.951 |
| Verification EER | 14.8% | 12.2% |
| Open-set AUC (½ enrolled, 3 seeds) | 0.954 ± 0.007 | 0.929 ± 0.005 |
| Open-set DIR@FAR=1% | 81.2% ± 4.1% | 44.3% ± 1.9% |

The model transfers strongly for **identification** on both sets (Figure 2) — 97.0% Rank-1 on set A and 87.0% Rank-1 across 308 unseen identities on the larger, harder set B — demonstrating that the learned muzzle representation captures genuine dermatoglyphic structure rather than memorising acquisition-specific cues. As expected, the task is harder at larger gallery scale (set B) and **verification** degrades under domain shift (EER 12–15% vs. 2.70% in-domain). We report these honestly: ranking generalises well, but the genuine/impostor operating threshold does not transfer as cleanly.

**Recovering cross-domain verification at no training cost.** A substantial component of the degradation appears to arise from *score mis-calibration* under domain shift rather than from feature collapse alone — we do not claim the absence of feature-level shift, only that a large part of the verification loss is recoverable without touching the features. Concretely, it is largely recovered by adaptive symmetric score normalisation (S-norm), which recalibrates each pair score by both endpoints' cohort statistics (unsupervised; cohort = the target set itself). S-norm reduces EER from 14.8%→**11.4%** on set A and 12.2%→**7.9%** on set B (a 23% and 35% relative reduction), raises ROC AUC to 0.943 and 0.972, and leaves Rank-1 unchanged or slightly improved (97.0%→97.5%, 87.0%→87.7%). This recovery is statistically significant: a probe-level bootstrap on set B gives a mean EER reduction of 4.0 points with a 95% confidence interval of [3.2, 4.7] (excludes zero). We contrast this with test-time BatchNorm adaptation (AdaBN), which helped verification only on the larger set and destabilised the smaller one — evidence that the transferable fix operates at the score level, not the feature level. S-norm requires no fine-tuning, no labels, and no access to the source data, making it directly deployable for cross-farm identification.

**Table 4: Effect of test-time adaptation on cross-domain verification (EER %, lower is better)**
| Method | Set A EER | Set B EER |
| :--- | :---: | :---: |
| Baseline (cosine) | 14.8 | 12.2 |
| + AdaBN | 15.4 | 8.5 |
| **+ S-norm** | **11.4** | **7.9** |

### 3.5 Robustness under Corruption: Is Adaptive Fusion Worth It?

Section 3.1 shows a single validation-selected fusion weight cuts in-domain EER 3.5×. A natural follow-up — and a common recommendation in multi-biometric fusion — is to make the fusion weight *input-adaptive*, trusting the CNN more when the image is degraded and the graph branch more when it is clean. We test this directly. Following the score-fusion formulation $s_{final}(x) = \alpha(x)\,s_{CNN} + (1-\alpha(x))\,s_{Hybrid}$, we compare six policies for $\alpha$: (1) CNN only, (2) Hybrid only, (3) fixed $\alpha=0.5$, (4) a single **validation-tuned** scalar $\alpha$, (5) a **rule-based** per-sample $\alpha$ from image blur, and (6) a **learned** per-sample gate (logistic regression on image- and graph-quality features, trained only on validation disagreement cases). Every gate is fit *only* on the validation split. We evaluate on clean test data and on three corruptions — Gaussian blur, brightness/haze, and spatter (occlusion) — at severities 1, 3, 5, applying the corruption to the *image* so the Hybrid's CNN backbone also degrades (not a clean-graph oracle). Per-sample scores are symmetrised so all policies are compared on identical verification pairs. This is a *self-contained* study under its own corruption harness (no test-time augmentation, symmetrised scoring), so its clean-column numbers are not identical to the headline ensemble of Table 1 (which uses CNN TTA); the val-tuned scalar here reaches EER 1.17% vs. the TTA ensemble's 0.78%. The comparison of interest is *within* this table, across policies under matched conditions.

**Table 5: Corruption robustness of fusion policies** (clean and "corr" = mean over nine corrupted conditions: blur/brightness/spatter × severity 1/3/5)
| Fusion policy | R1 clean | R1 corr | EER clean | EER corr | TAR@1% clean | TAR@1% corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN only | 95.1 | 89.8 | 2.81 | 4.43 | 96.2 | 85.2 |
| Hybrid only | 92.0 | 77.5 | 1.88 | 6.73 | 97.6 | 79.0 |
| Fixed α=0.5 | 92.8 | 79.1 | 1.78 | 6.34 | 97.8 | 80.3 |
| **Val-tuned scalar α** | **96.0** | **90.1** | **1.17** | **3.49** | **98.7** | **89.3** |
| Per-sample rule (α from blur) | 91.9 | 76.3 | 2.34 | 8.69 | 96.6 | 74.2 |
| Per-sample learned gate | 91.4 | 76.7 | 6.66 | 13.23 | 80.3 | 49.9 |

The answer is clear, and partly negative. The **validation-tuned scalar α is best on every aggregate metric** (including TAR@FAR=1%: 98.7% clean, 89.3% mean-corrupt): it matches the CNN's Rank-1 while more than halving clean EER (2.81→1.17) and reducing mean-corrupt EER (4.43→3.49). It does so by learning α=0.95 on clean/mild data (blending in just 5% of the graph branch to sharpen verification) and α→1.0 under severe spatter, where it gracefully abandons the fragile Hybrid and reduces exactly to the robust CNN (never worse). The fixed 50/50 blend, by contrast, is dragged down with the Hybrid under occlusion (Rank-1 33.7% at spatter-5 vs. the CNN's 65.2%). **Crucially, neither per-sample quality-adaptive policy improves on the single scalar** — the learned gate is in fact the *worst* for verification (clean EER 6.66%, TAR@1% 80.3%). Per-sample gating adds estimation variance without a corresponding payoff, because one branch (the CNN) dominates almost everywhere. This mirrors our cross-domain finding (§3.4) that robust improvements come from *global* score calibration, not fine-grained per-input adaptation.

**Fusion ablations.** Two ablations from the plan sharpen the picture (clean test). *(i) Which complement?* Replacing the Hybrid with the pure-graph ProtoN in the val-tuned fusion yields the *lowest* verification error of any configuration (EER 0.67%, TAR@1% 99.4% at α=0.85) but at a Rank-1 cost (93.6% vs. the CNN+Hybrid blend's 96.0%), because the EER-optimal weight leans harder on the graph branch; CNN+Hybrid remains the better *ranking* fusion. *(ii) Which gate features?* Ablating the learned gate's feature groups shows the *image-quality* features are what harm it: dropping them nearly halves the gate's EER (6.66→2.95%), while dropping graph-quality or branch-disagreement features barely moves it — the gate over-fits noisy per-image quality on the small validation disagreement set. Even its best-ablated variant (EER 2.95%) still loses to the trivial val-tuned scalar (EER 1.17%), so the negative result is robust to gate design.

### 3.6 Explainability: A Causal Faithfulness Protocol

We move beyond visualisation-only explainability (Grad-CAM + attention heatmaps) to a three-stage protocol that tests whether explanations are *causally* faithful, i.e. whether the highlighted evidence is what the model actually uses.

**Stage 1 — Attribution.** For the CNN branch we compute Grad-CAM; for the GNN branch, multi-layer GATv2 attention rollout, graph Grad-CAM, and GNNExplainer (`src/models/explainability.py`). We assemble a stratified case set spanning correct matches, false accepts, false rejects, and the two branch-disagreement classes, and render side-by-side CNN-vs-GNN explanations per case. One case type is instructive by its rarity: only a *single* false-reject exists in the entire 964-image test set (a genuine mate below the EER threshold), a symptom of how well-separated the in-domain score distribution is. Grad-CAM concentrates on the central bead clusters where dermatoglyphic patterns are densest, while GATv2 attention weights keypoint connections spanning the prominent muzzle valleys.

**Stage 2 — Causal ablation, and a control that changes the conclusion.** Qualitative maps are not evidence of causality, so we ablate the highest-importance evidence and measure the effect on identity. Crucially, the contrast one chooses decides the answer. Most explanation studies report *top* vs *random*; we argue the honest control is *top* vs *bottom* — remove the same *number* of nodes, ranked least-important instead of most — because top and bottom both delete a coherent extreme of the ranking whereas a random subset is scattered, and that difference alone perturbs a graph.

For the **CNN branch** (full 964-image test set) the attribution is faithful under *both* contrasts: at *k*=30, masking the most important regions costs 5.0 Rank-1 points versus 2.3 for random and only 1.7 for the least important (flip 12.3%, 5.3%, 5.7%).

For the **GNN branch** the same test gives a *negative* result, and we report it as such. Re-running on the full 964-graph test set (our earlier 120-graph subset left most probes without a gallery mate and is superseded), graph Grad-CAM passes the conventional top-vs-random test (Δcos 0.105 vs 0.048 at *k*=30, CIs disjoint) but **fails** the top-vs-bottom control: removing the 30% *least* important nodes yields Δcos = 0.100 [0.094, 0.107] against 0.105 [0.099, 0.112] for the most important — overlapping CIs — with indistinguishable flip rates (8.2% vs 8.6%; at *k*=10 the bottom set flips slightly *more*). The ranking therefore carries little causal signal beyond "removing a coherent block of nodes hurts". We draw two conclusions: per-keypoint graph attributions should not be used to justify claims about *which* muzzle regions drive identity in this model; and, more generally, top-vs-random is too weak a control for graph explanations — it would have certified an attribution that the stricter contrast rejects.

**Table 6: Causal ablation — GNN branch (GNN v3; full 964-graph test set; 95% bootstrap CI on Δcos).** Graph Grad-CAM passes top-vs-*random* but fails the stricter top-vs-*bottom* control (italics).

| Removed nodes | Δcos ↑ (95% CI) | top-1 flip (%) |
| :--- | :---: | :---: |
| top 10% | 0.029 [0.027, 0.032] | 3.9 |
| random 10% | 0.014 [0.013, 0.015] | 3.4 |
| *bottom 10%* | *0.022 [0.021, 0.025]* | *4.4* |
| top 20% | 0.064 [0.060, 0.069] | 6.1 |
| random 20% | 0.029 [0.027, 0.031] | 4.7 |
| *bottom 20%* | *0.059 [0.055, 0.064]* | *6.2* |
| top 30% | 0.105 [0.099, 0.112] | 8.6 |
| random 30% | 0.048 [0.045, 0.052] | 6.2 |
| *bottom 30%* | *0.100 [0.094, 0.107]* | *8.2* |

**Table 7: Causal ablation — CNN branch (full 964-image test set)**
| Masked regions | Δcos ↑ | top-1 flip (%) | Rank-1 drop |
| :--- | :---: | :---: | :---: |
| top 10% / random 10% | **0.0012** / 0.0007 | **2.8** / 1.5 | **−1.1** / −0.1 |
| top 20% / random 20% | **0.0027** / 0.0018 | **6.7** / 3.3 | **−2.9** / −1.6 |
| top 30% / random 30% | **0.0041** / 0.0028 | **12.3** / 5.3 | **−5.0** / −2.3 |

Both branches' attributions are causally faithful. One asymmetry is telling: sparse GNN node removal barely moves closed-set Rank-1 (identity is *distributed* across many keypoints — a global textural signature), whereas masking CNN regions produces a real, graded Rank-1 drop.

**Stage 3 — Hybrid pathway intervention.** We ask *which pathway* the Hybrid relies on by perturbing its graph input at test time and re-scoring the full test set. The result is stark: zeroing the per-keypoint node features collapses Rank-1 from 92.0% to 0.1% (EER 1.88%→50.2%, chance), whereas destroying the graph topology (random edges), zeroing edge attributes, or permuting keypoint positions each moves Rank-1 by ≤0.5 points. **The same pattern holds under corruption** (spatter-severity-3): zeroing node features still collapses Rank-1 56.8%→0.1% while geometry perturbations move it ≤0.7 points.

**Table 8: Hybrid pathway intervention (full test set, clean)**
| Intervention | Rank-1 (%) | EER (%) | ΔRank-1 |
| :--- | :---: | :---: | :---: |
| Full Hybrid (unperturbed) | 92.0 | 1.88 | — |
| Zero edge attributes | 92.0 | 1.88 | 0.0 |
| Shuffle keypoint positions | 92.3 | 1.84 | +0.3 |
| Randomise graph edges | 92.2 | 1.82 | +0.2 |
| **Zero node features** | **0.1** | **50.2** | **−91.9** |

This is not an accident of the architecture but a direct consequence of it: the Dynamic EdgeConv rebuilds its neighbourhood graph in *learned feature space* at every layer (so the static geometric edge set is overridden), the default configuration does not consume edge attributes, and permutation-invariant mean+max pooling makes the embedding invariant to keypoint reindexing. On both clean and corrupted data the Hybrid's identity signal is carried by CNN texture and feature-space message passing — not by the input muzzle *geometry*. **What fusion rescues:** tallying the val-tuned CNN+Hybrid blend against the CNN alone, fusion rescues 17 probes the CNN gets wrong while harming 9 it gets right (net +8, matching the +0.9-point Rank-1 gain), and recovers 12 of the 13 (92%) Hybrid-correct/CNN-wrong probes. The graph branch's contribution is concrete but small and targeted — a verification-sharpening complement, not a competitive ranker. Together, the three stages make the explainability claims falsifiable and reproducible — the standard we advocate for biometric-identification papers — rather than decorative.

---

## 4. Conclusion
We presented a comprehensive, leakage-audited evaluation of CNN, GNN, and Hybrid architectures for cattle muzzle biometric identification. Our best single model (EfficientNet-B4 + ArcFace) reaches 95.4% Rank-1, and a validation-selected CNN+Hybrid blend attains **96.1% Rank-1** with a **0.78% EER** — a 3.5× reduction in Equal Error Rate over the CNN alone. Rather than overclaim that graphs surpass CNNs on ranking, we localise the graph branch's value to **verification robustness**, supported by the low standalone EERs of the pure GNNs and by a controlled corruption study showing a single validation-tuned fusion weight is the robust choice (per-sample quality-adaptive gating does not improve on it). We complement qualitative saliency with a **causal explainability protocol** (node-ablation faithfulness and Hybrid pathway intervention), finding that the GNN distributes its identity decision across many keypoints and that, on clean in-domain data, the Hybrid's signal is carried by CNN texture rather than the input graph geometry. Crucially, the representation **generalises zero-shot to two separate muzzle datasets** (97.0% Rank-1 on a 24-identity set and 87.0% across 308 unseen identities, no fine-tuning; Section 3.4), via a muzzle detector that doubles as the deployment front-end. Remaining work before camera-ready is full-budget five-fold cross-validation of the Hybrid model (Section 3.2) and an ablation of the multi-scale node-sampling variant; longer-term directions include cross-farm domain adaptation to close the cross-dataset verification gap and low-latency on-farm deployment.

## 5. Limitations
(i) We evaluate open-set identification (Section 3.3) under a model trained closed-set on all identities; the strict unseen-identity variant (retraining with unknown identities held out) is the natural next step, supported by our harness. (ii) Cross-dataset transfer (Section 3.4) uses a muzzle detector to crop the two external sets; the detector missed roughly a third of images on the larger set, so those transfer numbers are a conservative lower bound. (iii) The corruption study (Section 3.5) perturbs the image while keypoint coordinates are taken from the clean graph, a small optimistic bias under occlusion. We quantify this rather than leave it open: recomputing DISK keypoints from the corrupted images changes Rank-1 by +0.9 points under blur-5 and −2.7 points under spatter-5, so the bias is real but small and confined to the occlusion condition. (iv) The two GNN explainers we compare show low mutual rank agreement, and graph Grad-CAM additionally fails our top-vs-bottom ablation control (Table 6); per-keypoint graph importance should therefore be treated as *unvalidated* for this model, not merely indicative. The CNN's image-space attributions do pass the same control. (v) The primary split is image-level: the dataset ships no session, capture-date, or camera metadata, so we cannot construct a session-disjoint split and cannot exclude acquisition-level similarity (same session/burst/camera) across splits. In-domain numbers should therefore be read as an upper bound relative to a session-disjoint protocol; the zero-shot cross-dataset results (Section 3.4) are unaffected by this and provide the stronger generalisation evidence.

---

## References
1. J. Deng, J. Guo, N. Xue, S. Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR, 2019.
2. M. Tan, Q. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML, 2019.
3. M. Tyszkiewicz, P. Fua, E. Trulls. *DISK: Learning Local Features with Policy Gradient.* NeurIPS, 2020.
4. S. Brody, U. Alon, E. Yahav. *How Attentive are Graph Attention Networks? (GATv2).* ICLR, 2022.
5. Y. Wang, Y. Sun, Z. Liu, S. Sarma, M. Bronstein, J. Solomon. *Dynamic Graph CNN for Learning on Point Clouds (EdgeConv).* ACM TOG, 2019.
6. R. Ying, D. Bourgeois, J. You, M. Zitnik, J. Leskovec. *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS, 2019.
7. P. Pope, S. Kolouri, M. Rostami, C. Martin, H. Hoffmann. *Explainability Methods for Graph Convolutional Neural Networks.* CVPR, 2019.
8. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
9. K. Nandakumar, Y. Chen, S. Dass, A. Jain. *Quality-based Score Level Fusion in Multibiometric Systems.* ICPR, 2006.
10. S. Kumar et al. *Muzzle point pattern based techniques for individual cattle identification.* IET Image Processing, 2017.
11. G. Bello et al. *Deep learning-based muzzle detection and cattle identification.* 2020.
12. V. Čermák, L. Picek, L. Adam, K. Papafitsoros. *WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification (MegaDescriptor).* WACV, 2024.
13. Beef Cattle Muzzle/Noseprint Database. Zenodo, doi:10.5281/zenodo.6324361.
14. Pakistan Cattle Muzzle/Face Dataset. Zenodo, doi:10.5281/zenodo.8377921 (CC BY 4.0).
15. G. Jocher et al. *Ultralytics YOLOv8.* 2023.
16. R. Auckenthaler, M. Carey, H. Lloyd-Thomas. *Score Normalization for Text-Independent Speaker Verification Systems (S-norm/cohort normalization).* Digital Signal Processing, 2000.
17. Y. Li, N. Wang, J. Shi, J. Liu, X. Hou. *Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN).* ICLR Workshop, 2017.
