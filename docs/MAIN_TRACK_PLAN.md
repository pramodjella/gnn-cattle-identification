# Main-Track Research Plan — Label-Free Test-Time Score Calibration for Cross-Domain Re-ID

**Status: feasibility CONFIRMED (2026-07-07).** This is a distinct paper from the cattle
domain paper; the cattle work becomes one supporting case study.

## Thesis (novel, testable)
Cross-domain degradation in biometric re-identification is primarily a **score-distribution
mis-calibration** problem, not a feature problem — and it is recoverable **label-free, without
retraining**, by adaptive score normalisation. This is **modality-agnostic** (muzzle ridges,
faces, coat patterns) and **model-agnostic** (our CNN and the SOTA foundation model MegaDescriptor).

## Evidence so far (`scripts/wildlife_probe.py`)
Cross-domain protocol = clean gallery vs corrupted probes; corruption = controlled domain shift.

| Model / data | Modality | Shift | Baseline EER | +S-norm EER | Recovery |
| :-- | :-- | :-- | :--: | :--: | :--: |
| Cattle CNN (ours) | muzzle | cross-dataset A | 14.8% | 11.4% | −23% |
| Cattle CNN (ours) | muzzle | cross-dataset B | 12.2% | 7.9% | −35% |
| MegaDescriptor-L | macaque face | spatter | 6.24% | 5.34% | −14% |
| MegaDescriptor-L | Friesian coat | spatter | 2.77% | 1.15% | **−59%** |

Consistent direction, never harmful under no-shift. AdaBN (feature-level) is unreliable — the
fix lives at the **score** level.

## Novelty status: SETTLED with statistical rigor — no mechanism significantly beats S-norm (2026-07-07)
Across 3 datasets/modalities (Friesian coat, macaque face, IPanda50 pattern) on MegaDescriptor,
quality-conditioned S-norm appeared to win a few cases by ~0.1-0.2% EER — but **bootstrap 95% CIs
on the S-norm vs quality-snorm EER difference straddle zero** (e.g. IPanda spatter-3: +0.03% CI
[-0.51,+0.50] -> n.s.; spatter-5: -0.07% CI [-0.49,+0.56] -> n.s.). The apparent edge is NOISE.
Definitive: **no calibration mechanism beats plain S-norm at a significant level.** The novelty
question is closed; pursue the EMPIRICAL paper (S-norm robustly recovers cross-domain EER across
species/modalities/backbone), not a method paper. A real method contribution needs a fundamentally
different idea (learned calibrator + theory / joint feature-score adaptation) — open research.

## (superseded) Novelty status: TESTED — no mechanism reliably beats S-norm (2026-07-07)
Bake-off (`scripts/calibration_bakeoff.py`, `src/evaluation/calibration.py`) of 5 mechanisms —
S-norm, AS-norm(top-k), quantile/rank-norm, quality-conditioned S-norm, source-Align — across 2
datasets (Friesian coat, macaque face) x multiple shift severities on MegaDescriptor:
- Friesian: quality-snorm wins spatter-3; s-norm wins spatter-5; as-norm/align LOSE.
- Macaque: as-norm(k=40) wins brightness-3 & spatter-3; quality-snorm marginally wins spatter-5.
=> Winner is INCONSISTENT across datasets, margins small, S-norm always close. NO method dominates.
**Conclusion: the CVPR-main method-novelty bar is NOT met with these ideas.** Do not overfit a
"winner" to one dataset. Realistic target = strong IJCB/WACV **empirical** paper: "label-free score
calibration robustly recovers cross-domain re-ID across modalities and the SOTA foundation model."
A genuine method contribution would need a fundamentally different idea (e.g., learned calibrator with
theory, or joint feature+score adaptation) — open research, uncertain payoff.

## (superseded) Novelty status: OPEN (first attempt failed)
Severe shift gives a REAL gap: MegaDescriptor on FriesianCattle2017 spatter-5 = **27.67% EER**,
S-norm recovers to 22.81% (-18%). BUT the first novel mechanism — **source-distribution Align**
(per-probe Z-norm rescaled to clean gallery-gallery impostor stats) — LOST to plain S-norm
(26.50% vs 22.81%). So the empirical finding (S-norm recovers cross-domain EER) is confirmed and
strong, but a *novel method that beats S-norm* is not yet demonstrated. This is the make-or-break
for CVPR-main vs IJCB/WACV. Next mechanism ideas to try: adaptive top-k cohort (AS-norm),
quality/entropy-conditioned calibration strength, learned unsupervised calibrator, higher-moment
distribution matching. If none beats S-norm, position as a rigorous cross-modality empirical study
(IJCB/WACV), not a method paper.

## What is still needed for a main-track submission
1. **A dataset/split where MegaDescriptor genuinely degrades (large baseline gap).** The controlled
   corruption shift is a proxy; the strongest evidence is a REAL cross-dataset / held-out-distribution
   shift where MegaDescriptor drops many points (the literature reports up to −36 top-1). Use
   WildlifeReID-10k's time/similarity-aware splits, or gallery-from-dataset-A / probe-from-dataset-B
   of the same species.
2. **A NOVEL calibration mechanism beyond vanilla S-norm** (the novelty bar). Candidates:
   quality-conditioned calibration (gate on per-sample quality), distribution-alignment calibration,
   or a learned unsupervised target calibrator. Needs a theoretical argument for *why* score
   calibration transfers where feature adaptation does not.
3. **Breadth:** ≥5 datasets / multiple species; ≥2 backbones (MegaDescriptor, DINOv2).
4. **Baselines:** TENT (ICLR'21), TDA (CVPR'24), DART³ (re-ID TTA), TAF-Cal — beat/match them
   at zero label cost.
5. **Open-set + cross-domain** jointly (the field's hardest, most-cited regime).

## Immediate next experiments (cheap, high-information)
- Run the probe on 3–4 more `wildlife_datasets` incl. a set MegaDescriptor likely did NOT train on
  (bigger baseline gap → clearer S-norm win).
- Real cross-dataset shift: two datasets of the same species, gallery A / probe B.
- Prototype one novelty mechanism (quality-conditioned calibration) and compare to plain S-norm.

## Honesty guardrails (lessons from this project)
- Muzzle print does NOT exist in wildlife — cattle stays domestic; the METHOD carries, not the modality.
- Always verify a suspiciously good number in a second environment (the Modal hybrid inflation).
- Watch for oracle/duplicate confounds (clean-graph fusion; OriginalMaster pool).
