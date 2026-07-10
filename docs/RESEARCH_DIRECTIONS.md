# Research Directions: Improving / Innovating Beyond the Current Results

Grounded in this project's verified findings + a literature scan (2024-2025). Confidence levels
and risks are honest.

## The core reframe (high confidence, low risk) — GNNs belong at RE-RANKING, not feature extraction
The project's GNN was placed where it was doomed: **feature extraction**, competing with CNNs /
foundation models (it lost, 92% vs 95% Rank-1). The re-ID literature is clear that GNNs add value at
the **gallery-graph / re-ranking / test-time** stage — relational reasoning over the probe-gallery
structure (Deep Similarity-Guided GNN, arXiv:1807.09975; Graph Neural Re-Ranking, arXiv:2406.11720).
That is exactly where our verified "cross-domain = score mis-calibration" finding lives. Reframing the
GNN from failed feature-competitor to **relational test-time calibrator** turns the weakness into the
contribution.

## Novel method proposal: GraphCal — a learned, label-free, test-time GNN score calibrator
**Idea.** A GNN over the target probe-gallery similarity graph. Node features = cohort statistics
(S-norm-like: local score mean/std/rank). Edge features = reciprocal-neighbour consistency
(k-reciprocal-like). The GNN message-passes to output recalibrated pairwise scores. **Meta-trained
across many SOURCE datasets** (WildlifeReID-10k bundles 37) with an episodic domain-generalization
objective, so it generalizes to unseen target domains and runs **label-free** at deployment.

**Why novel.** Unifies distributional calibration (S-norm) + topological re-ranking (k-reciprocal)
into ONE *learned* module, meta-trained for cross-domain generalization, motivated by our proven
calibration-not-features finding. Model-agnostic (operates on any backbone's score graph:
MegaDescriptor, DINOv2). The pieces exist separately — reciprocal re-ranking (Zhong 2017), GNN
re-ranking (2024), meta-DG on graphs (MLDGG 2024), test-time DG via graph matching (CVPR 2025) — but
not this synthesis for re-ID score calibration. That CVPR 2025 has a related test-time-DG graph paper
means the area is main-track-active (real ceiling, real bar).

**DE-RISK RESULT (2026-07-07): complementary but MODALITY-DEPENDENT** (`scripts/rerank_experiment.py`,
`src/evaluation/rerank.py`, DINOv2 cross-domain):
- MacaqueFaces (face): S-norm 21.9% -> k-recip->S-norm **19.3%** (combo -2.5pt). COMPLEMENTARY.
- CZoo (chimp face): S-norm 14.5% -> combo **14.0%** (-0.5pt). marginal.
- IPanda50 (panda pattern): S-norm 35.5% -> combo 36.0% (WORSE); k-reciprocal even hurts baseline.
  NOT complementary.
=> Same pattern as quality-conditioning: topological/auxiliary signals help FACES but not repetitive
PATTERNS (k-recip mutual-neighbour structure is informative for faces, noisy for coats). The
hand-crafted combo does NOT robustly beat S-norm. **This is the argument FOR GraphCal:** a learned
GNN can learn WHEN topology is trustworthy (faces) vs not (patterns) and gate it per-input — which
fixed k-reciprocal cannot. The modality-dependence is the central challenge/story, not a clean win.
Also a standalone honest finding: "the value of topology/quality signals for test-time re-ID
calibration is modality-dependent."

**Plan.** (1) confirm complementarity on 3+ datasets/backbones; (2) build GraphCal (reuse project's
GNN stack: edge_conv, TRM, adaptive_graph); (3) meta-train on WildlifeReID-10k source split, evaluate
on held-out datasets; (4) baselines: S-norm, k-reciprocal, RRE re-ranking, AdaBN, TENT; (5) show it
beats the best hand-crafted combo and generalizes to unseen domains label-free.

**Risk.** Medium-high. Meta-training to generalize to UNSEEN domains is hard; must beat strong
hand-crafted baselines (S-norm + k-reciprocal combo, which is already good). If it only ties the
combo, it is not novel enough — but the combo itself is then a small contribution.

## Secondary directions (lower priority, grounded)
1. **Conformal test-time calibration.** Frame the verification threshold as conformal prediction to
   get distribution-free coverage guarantees on genuine/impostor decisions under shift. Novel for
   re-ID; theoretically clean; complements S-norm.
2. **Learned quality gate (salvage our negative).** We found quality-conditioned calibration helps
   faces but hurts coat patterns. A small learned gate (per-sample, from image/graph quality) that
   decides WHEN to apply quality conditioning could turn that modality-dependent negative into a
   positive, and is a natural node-feature inside GraphCal.
3. **MLLM re-ranking** (arXiv:2606.16161-style). Trendy, heavy; lower priority for a lean lab.

## What NOT to pursue (evidence-based)
- More hand-crafted score-norm variants (AS-norm/quantile/quality) as a "method" — bootstrap-tested,
  none robustly beats plain S-norm.
- GNN as a feature extractor to beat foundation models — it loses; wrong role for the GNN.
- Modal for these probes — light enough for local; Modal inflated the hybrid eval (untrustworthy).
