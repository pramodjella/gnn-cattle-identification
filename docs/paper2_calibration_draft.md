# Draft 2 (main-track attempt): When Does Label-Free Score Calibration Recover Cross-Domain Animal Re-Identification?

**Venue target:** IJCB / WACV (empirical + analysis paper). Distinct from the cattle domain paper.
**Status:** evidence assembled; skeleton draft. Novelty is empirical/analytical, not a new method.

---

## Abstract (draft)
Foundation models for animal re-identification (e.g. MegaDescriptor) achieve near-perfect
verification in-distribution but degrade under domain shift — different farms, sessions, or
capture conditions. We show, across three backbones (a task-specific CNN, MegaDescriptor, and the
general-purpose DINOv2), four biometric modalities (cattle muzzle, primate face, cattle coat,
panda pattern), and three shift types (cross-dataset, synthetic corruption, and natural
cross-session splits), that this degradation is primarily a **score mis-calibration** problem
rather than a feature problem, and is partially recoverable **without any labels, retraining, or
source data** via adaptive symmetric score normalisation (S-norm). Crucially, we characterise
*when* it helps: the recovery is significant precisely when the score distribution is
mis-calibrated — weak/general backbones, or strong backbones under severe shift — and negligible
when a strong backbone faces only mild shift. Feature-level test-time adaptation (AdaBN) is by
contrast unreliable. Our analysis reframes cross-domain re-ID robustness as a calibration question
and provides a deployable, zero-cost baseline that practitioners can always apply.

## 1. Introduction
- Animal re-ID moved to foundation models; the field's #1 open problem is cross-domain / open-set
  generalisation (MegaDescriptor drops up to 36 top-1 cross-domain).
- Test-time adaptation is active but mostly feature-level and often needs adaptation steps.
- Our finding: the transferable, zero-cost fix is at the SCORE level, and its benefit is
  *predictable* from a measurable mis-calibration signal.
- Contributions: (i) a broad, honest empirical demonstration across models/modalities/shifts;
  (ii) a conditional characterisation (benefit scales with mis-calibration) with significance
  testing; (iii) evidence that score-level beats feature-level (AdaBN) adaptation for this problem.

## 2. Method
- Adaptive symmetric S-norm on the probe×gallery score matrix (unsupervised, cohort = target set).
- Mis-calibration indicator (baseline EER / score-distribution spread) as the predictor of benefit.
- Contrast: AdaBN (feature BN adaptation) and quality-conditioned / AS-norm / quantile variants
  (none significantly beat plain S-norm — reported honestly).

## 3. Experiments
**Table 1: S-norm recovery across models, modalities, shifts (EER, bootstrap 95% CI).**
| Backbone | Modality | Shift | Baseline EER | +S-norm | Recovery | Significant? |
| :-- | :-- | :-- | :--: | :--: | :--: | :--: |
| Task CNN (ours) | cattle muzzle | cross-dataset B (308 IDs) | 12.2% | 7.9% | −35% (+4.0pt) | **yes** [+3.2,+4.7] |
| MegaDescriptor-L | macaque face | corruption (spatter) | 6.24% | 5.34% | −14% | (large-EER) |
| MegaDescriptor-L | Friesian coat | corruption (spatter) | 2.77% | 1.15% | −59% | (large-EER) |
| MegaDescriptor-L | panda pattern | natural cross-video | 2.63% | 2.29% | −13% | **n.s.** |
| DINOv2 (general) | panda pattern | natural cross-video | 40.8% | 35.3% | −13% | **yes** |
| DINOv2 (general) | panda pattern | in-domain (random) | 38.0% | 31.4% | −17% | **yes** |

## 4. Analysis — the conditional law
- Recovery magnitude and significance scale with baseline mis-calibration (EER / score spread).
- Well-calibrated strong backbone + mild shift => nothing to fix (honest negative).
- Weak/general backbone or severe shift => significant, sizeable recovery.
- Score-level (S-norm) transfers; feature-level (AdaBN) does not reliably.

## 5. Limitations (honest)
- No proposed mechanism significantly beats plain S-norm (AS-norm / quantile / quality-conditioned
  all tested; bootstrap n.s.). This is an empirical/analysis contribution, not a new method.
- Corruption shift is a controlled proxy; natural shifts are limited to available metadata.
- Cattle muzzle biometrics does not exist in wildlife; only the calibration *method* transfers.

## References
MegaDescriptor (WACV'24); DINOv2; TENT (ICLR'21); TDA (CVPR'24); S-norm (Auckenthaler 2000);
AdaBN (Li 2017); WildlifeReID-10k; Nandakumar quality fusion (ICPR'06).
