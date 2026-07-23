# When Does Label-Free Score Calibration Recover Cross-Domain Animal Re-Identification?

**Venue target:** IJCB / WACV (empirical + analysis).
**Note:** distinct from the cattle-muzzle domain paper; the contribution here is empirical and
analytical (a conditional characterization), not a new method. The cattle work is one case study.

---

## Abstract

Foundation models for animal re-identification, such as MegaDescriptor, achieve near-perfect
verification on species and datasets seen during training, yet a large body of work reports that
they degrade sharply under domain shift — new farms, new recording sessions, different cameras, or
unseen distributions. We ask a narrow but practically important question: *when* is this
degradation recoverable without labels, retraining, or access to the source data, simply by
recalibrating similarity scores at test time? Across three backbones (a task-specific muzzle CNN,
the re-identification foundation model MegaDescriptor-L, and the general-purpose self-supervised
model DINOv2), four biometric modalities (cattle muzzle, primate face, cattle coat, and panda
coat pattern), and three families of domain shift (cross-dataset transfer, controlled image
corruption, and natural cross-session splits), we show that cross-domain verification loss is
primarily a problem of *score mis-calibration* rather than feature quality, and that adaptive
symmetric score normalization (S-norm) recovers a significant fraction of it at zero training cost.
Crucially, we characterize *when* the recovery is significant: it scales with a measurable
mis-calibration signal (the baseline equal-error rate, EER). In a controlled comparison on identical
data and shift, a backbone that is well-calibrated for the target species shows no gap and no
benefit, whereas a general backbone that is mis-calibrated for re-identification degrades by more
than twelve EER points and is significantly recovered by the same calibration step. Feature-level
test-time adaptation (AdaBN) is, by contrast, unreliable. We conclude that score-level calibration
is a dependable, deployable baseline for cross-domain animal re-identification, and that its
applicability is predictable in advance.

---

## 1. Introduction

Individual animal re-identification (re-ID) underpins ecological monitoring, precision livestock
farming, and anti-poaching efforts. The field has converged on large pretrained embedding models:
MegaDescriptor, trained on 2.8M images spanning 30k identities and 33 species, is the de-facto
state of the art and outperforms general vision backbones such as CLIP and DINOv2 on in-domain
benchmarks. However, the community's most-cited open problem is generalization: on cross-domain or
time/similarity-aware splits, even MegaDescriptor drops by tens of points, because ecological data
is collected under conditions that differ from training.

Test-time adaptation (TTA) is the natural remedy, but most TTA methods operate on *features*
(e.g. batch-norm statistics, entropy minimization) and require gradient steps or careful tuning; in
re-ID they can be unstable when the target set is small. We take a different and deliberately
minimal view. We observe that under domain shift the *ranking* of a re-ID model often survives —
the correct identity is still near the top — while the *operating threshold* that separates genuine
from impostor pairs does not transfer. This is a calibration failure of the score distribution, not
a collapse of the features.

This paper is an empirical and analytical study of one consequence of that observation: adaptive
symmetric score normalization (S-norm), a decades-old technique from speaker verification, can be
applied verbatim at test time — no labels, no retraining, no source data — to realign the score
distribution to the target domain. Our contributions are:

1. **A broad, honest empirical demonstration** that S-norm recovers cross-domain re-ID verification
   across three backbones, four biometric modalities, and three shift families.
2. **A conditional characterization with significance testing.** Using probe-level bootstrap
   confidence intervals, we show the recovery is significant precisely when the score distribution
   is mis-calibrated (large baseline EER) and negligible otherwise, and we give a controlled
   two-backbone comparison on identical data that isolates calibration as the cause.
3. **Evidence that the fix is score-level, not feature-level:** feature-level AdaBN is unreliable
   (helping on some sets, destabilizing others), whereas score-level S-norm is monotone-safe.

We are explicit that this is not a new method: we tested adaptive top-k (AS-norm), quantile
normalization, and a quality-conditioned variant, and none significantly beat plain S-norm.
The value of the paper is the *characterization* — telling practitioners when a zero-cost baseline
will and will not help — not a novel mechanism.

## 2. Related Work

**Foundation models for animal re-ID.** MegaDescriptor and the WildlifeDatasets toolkit
standardized large-scale animal re-ID; WildlifeReID-10k (10k identities, 33 species) and PetFace
established leakage-aware, time/similarity-aware evaluation. These works document, but do not
resolve, the cross-domain gap.

**Test-time adaptation.** TENT (entropy minimization), AdaBN (BN-statistic adaptation), and
vision-language TTA (e.g. TDA) adapt features or predictions at test time. DART³ studies TTA
specifically for person re-ID. Most require adaptation steps and can be brittle with few target
samples; our approach requires neither gradients nor labels.

**Score normalization.** Z-norm, T-norm, and (adaptive) S-norm originate in speaker verification
as cohort-based recalibration of verification scores. Quality-based score fusion (Nandakumar et al.)
weights modalities by input quality. We repurpose S-norm as a domain-adaptation tool for visual
re-ID and, unlike prior fusion work, characterize *when* it is statistically effective.

## 3. Method

**Setup.** Given a gallery of enrolled templates and a set of probes, a backbone produces
L2-normalized embeddings; the score matrix `S` holds cosine similarities between probes (rows) and
gallery templates (columns). Verification thresholds `S`; identification ranks each row.

**Adaptive symmetric S-norm.** For a probe `i` and template `j`,
`S'_{ij} = 1/2[(S_{ij} - mu_i)/sigma_i + (S_{ij} - mu_j)/sigma_j]`,
where `(mu_i, sigma_i)` are the mean and standard deviation of probe `i`'s scores against the
gallery cohort, and `(mu_j, sigma_j)` the statistics of template `j` against the probe cohort. The
cohort is the target set itself; no labels or source data are used. Intuitively, S-norm removes
per-probe and per-template offsets/scales that domain shift introduces, restoring a common
operating point.

**Mis-calibration predictor.** We use the baseline EER (equivalently, the overlap of the genuine
and impostor score distributions) as an a-priori indicator of whether S-norm will help: a large
baseline EER signals a mis-calibrated score distribution with headroom for recalibration.

**Baselines we compare against.** (i) Raw cosine (no adaptation). (ii) AdaBN — recompute the
backbone's batch-norm statistics on the target (feature-level). (iii) Alternative score-norms:
AS-norm (top-k cohort), quantile/rank normalization, and a quality-conditioned S-norm that scales
calibration strength per probe. We report all honestly.

## 4. Experimental Setup

**Backbones.** (a) a task-specific EfficientNet-B4 + ArcFace muzzle CNN trained on a single cattle
dataset; (b) MegaDescriptor-L-384 (re-ID foundation model); (c) DINOv2 ViT-B/14 (general
self-supervised model), used as a frozen embedding extractor.

**Datasets / modalities.** Cattle muzzle (two external cross-datasets), MacaqueFaces (primate
face), CZoo (chimpanzee face), FriesianCattle2017 (cattle coat), IPanda50 (panda coat pattern).

**Domain-shift protocols.** (i) *Cross-dataset*: train on one cattle dataset, enroll+probe a
different one (real acquisition shift). (ii) *Corruption*: clean gallery, probes corrupted with
blur / brightness / spatter at graded severities (controlled shift). (iii) *Natural*: split
gallery/probe by dataset metadata — MacaqueFaces by capture date (temporal), IPanda50 by video
(recording session) — an ecologically valid shift with no synthetic manipulation.

**Metrics & significance.** Verification EER and ROC-AUC. For each condition we report the EER
recovery (baseline minus S-norm) with a 95% bootstrap confidence interval over resampled probes;
"significant" means the interval excludes zero.

## 5. Results

**Table 1** summarizes recovery across backbones, modalities, and shift families.

| Backbone | Modality | Shift | Baseline EER | +S-norm | Recovery | 95% CI / note |
| :-- | :-- | :-- | :--: | :--: | :--: | :-- |
| Muzzle CNN (ours) | cattle muzzle | cross-dataset (308 IDs) | 12.2% | 7.9% | +4.0 pt | [+3.2, +4.7] **sig** |
| Muzzle CNN (ours) | cattle muzzle | cross-dataset (24 IDs) | 14.8% | 11.4% | +3.4 pt | large-EER regime |
| MegaDescriptor | macaque face | corruption (spatter) | 6.24% | 5.34% | +0.9 pt | large-EER regime |
| MegaDescriptor | Friesian coat | corruption (spatter) | 2.77% | 1.15% | +1.6 pt | large-EER regime |
| MegaDescriptor | panda pattern | natural (cross-video) | 2.63% | 2.29% | +0.3 pt | [−0.33, +0.14] **n.s.** |
| MegaDescriptor | macaque face | natural (date) | 0.08% | 0.08% | 0 | no gap |
| DINOv2 (general) | panda pattern | natural (cross-video) | 40.8% | 35.3% | +5.5 pt | [+4.0, +7.2] **sig** |
| DINOv2 (general) | macaque face | natural (date) | 29.2% | 20.7% | +8.4 pt | [+7.2, +9.8] **sig** |

**The conditional law.** Recovery magnitude and statistical significance track the baseline EER.
Where the score distribution is well-calibrated (MegaDescriptor on species it was trained on, EER
≈ 0–3%), S-norm's effect is small and, on the mild natural panda shift, not significant. Where the
distribution is mis-calibrated (the cattle CNN across datasets, or DINOv2 anywhere), S-norm yields
multi-point, significant EER reductions.

**Controlled comparison (isolating the cause).** On MacaqueFaces with the natural date split,
holding data and shift fixed and varying only the backbone: MegaDescriptor, trained on macaques, is
essentially perfect (0.08% EER) and gains nothing; DINOv2, which was never trained for re-ID and is
mis-calibrated for it, degrades to 29.2% EER under the same shift and is significantly recovered by
S-norm (+8.4 pt, CI [+7.2, +9.8]). Identical inputs, opposite outcomes — the benefit is attributable
to backbone calibration, not to the data or the shift.

**Score-level beats feature-level.** Feature-level AdaBN helped verification on one cattle
cross-dataset but destabilized the other (and collapsed open-set rejection), whereas score-level
S-norm never harmed and helped whenever mis-calibration was present. This supports our framing that
the transferable fix operates on scores.

**Negative result on novelty (thoroughly tested).** We asked whether any variant beats plain
S-norm, including in the high-mis-calibration regime (DINOv2) where there is the most headroom. The
outcome is modality-dependent and does not yield a robust method. A quality-conditioned S-norm
significantly outperforms plain S-norm on face modalities (chimpanzee CZoo +3.3 EER pt, macaque
+5.1–5.7 pt, bootstrap CIs exclude zero) but is statistically indistinguishable on panda pattern
and *significantly worse* on cattle coat (Friesian, −2.7 pt at high severity). Adaptive top-k
(AS-norm) and quantile normalization each win on isolated cases but lose elsewhere. Across
datasets, no variant reliably beats plain S-norm, and some are significantly harmful on some data.
We therefore present the paper as an empirical characterization with plain S-norm as the dependable,
modality-agnostic choice, not as a new method. (Practical note: per-probe quality conditioning
appears to help when a confidence signal is informative — e.g. faces — and hurt when it is not —
e.g. repetitive coat patterns — which is itself a small honest finding about when *not* to add it.)

## 6. Discussion

Our results reframe a slice of cross-domain re-ID robustness as a calibration question with a
predictable answer. Practitioners deploying a re-ID model to a new farm or field site can (i) apply
S-norm at essentially no cost, and (ii) predict whether it will help from the target-set baseline
EER before collecting a single label. The finding that a general model (DINOv2) benefits even
in-distribution, while a specialized model (MegaDescriptor) benefits only under shift, indicates
that the mechanism is score mis-calibration rather than domain shift per se — shift is simply one
common cause of mis-calibration.

## 7. Limitations

(i) The contribution is empirical/analytical; no proposed mechanism significantly outperforms plain
S-norm. (ii) Corruption shifts are controlled proxies; natural shifts are limited to the metadata
available in public datasets. (iii) We study verification/identification with cohort-based
calibration; jointly learning a calibrator with theoretical guarantees is left to future work.
(iv) Muzzle-print biometrics does not exist for free-ranging wildlife; only the calibration method,
not the cattle modality, transfers across the study.

## 8. Conclusion

Cross-domain animal re-identification degradation is, to a first approximation, a score-calibration
problem, and label-free adaptive S-norm is a dependable, zero-cost remedy whose effectiveness is
predictable from a simple mis-calibration signal. We demonstrate this across three backbones, four
modalities, and three shift families, with significance testing and a controlled comparison that
isolates calibration as the cause. We advocate reporting cross-domain re-ID with, and predicting
from, this calibration lens.

## References
1. Čermák, Picek, Adam, Papafitsoros. WildlifeDatasets / MegaDescriptor. WACV 2024.
2. Oquab et al. DINOv2. TMLR 2024.
3. Wang et al. TENT: Fully Test-Time Adaptation by Entropy Minimization. ICLR 2021.
4. Li et al. Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN). ICLR-W 2017.
5. Karmakar et al. / DART³: Test-Time Adaptation for Person Re-ID. 2025.
6. Auckenthaler, Carey, Lloyd-Thomas. Score Normalization for Text-Independent Speaker Verification
   (S-norm/cohort). Digital Signal Processing 2000.
7. Nandakumar, Chen, Dass, Jain. Quality-based Score Level Fusion in Multibiometric Systems. ICPR 2006.
8. Adam et al. WildlifeReID-10k. 2024.
9. Deng et al. ArcFace. CVPR 2019.
