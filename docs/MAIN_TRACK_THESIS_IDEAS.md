# Main-Track Thesis — Better Ideas

*A working document to sharpen the main-track (calibration) paper into something with a
clear, novel, defensible thesis. Builds on `MAIN_TRACK_PLAN.md` and `RESEARCH_DIRECTIONS.md`.
Every claim below is checked against `outputs/stats/*.json`.*

---

## 0. DE-RISK OUTCOME (2026-07-24) — the modality-dependence thesis is REFUTED. Do not pursue it.

I ran the make-or-break experiment (`scripts/experiment_modality_law.py`, committed
`outputs/stats/modality_law_{megadescriptor,dinov2}.json`): for MacaqueFaces / CZoo / IPanda50,
per-identity-template enrollment, clean gallery vs spatter-3 probe, compute the label-free
`reciprocal_consistency` statistic and the structural-signal benefit (quality-snorm and
k-reciprocal vs plain S-norm) with probe-bootstrap CIs. **The thesis in §3 does not survive.**

**Findings (honest):**
- **The predicted face-advantage is absent / reversed.** On DINOv2 (the general backbone where
  a gap exists), quality-snorm significantly helps **IPanda (the PATTERN)** (+2.61 pt, CI
  [+0.75,+4.49] excludes 0), while on both FACES it is positive but **not significant** (+1.54,
  +1.51; CIs include 0). This is the *opposite* of "structural helps faces, not patterns."
- **The label-free statistic does not predict the benefit.** On DINOv2 it does not even separate
  faces from patterns (MacaqueFaces 0.170 < IPanda 0.192 < CZoo 0.220); consistency-vs-benefit
  Spearman is −0.5 (wrong sign).
- **On MegaDescriptor (specialized, ~3% face EER) there is no benefit anywhere** — no gap, nothing
  to calibrate. The effect is entangled with the conditional law and is backbone-conditional.
- **k-reciprocal hurts on every dataset/backbone** (−4.6 to −6.9 pt on DINOv2).
- Even the committed bakeoff it was built on doesn't reconcile: `calibration_bakeoff.json` called
  IPanda quality-snorm *n.s.*; my controlled run makes it the one *significant* win. The benefits
  are small (1–3 pt), protocol- and backbone-sensitive, and not modality-aligned.

**Verdict:** the "structural signals are modality-dependent, predicted by a reliability statistic"
story is not real enough to carry a paper. GraphCal (which this was meant to motivate) is
**not justified** — building it would have been weeks spent on a false premise. The fail-fast worked.

**What the data DOES robustly support (the pivot):** plain **S-norm recovers cross-domain EER on
every backbone and modality** (DINOv2: MacaqueFaces 26.4→15.1, CZoo 38.1→21.4, IPanda 44.3→38.7),
while structural add-ons (quality-snorm, k-reciprocal) give at best small, inconsistent gains and
k-reciprocal reliably *hurts*. → **Strengthen the EXISTING calibration paper's honest thesis**
(the conditional law + "nothing robustly beats plain S-norm") with this NEW committed multi-backbone
evidence, and resolve the §7 provenance blockers by regenerating those numbers (now that
`wildlife_natural_shift.py` persists JSON). The sections below are kept as a record of the
(refuted) exploration.

---

## 1. Where the current thesis stands, honestly

**Current thesis (paper2):** *cross-domain re-ID degradation is a score-calibration problem,
recoverable label-free by S-norm; recovery is significant iff a baseline gap exists.*

**Why it is not yet main-track:**
- It analyses an existing method (S-norm, Auckenthaler 2000). The conditional law reads as
  "S-norm works when there's something to fix" — a reviewer can call it expected.
- We proved no hand-crafted variant robustly beats S-norm *on patterns* — honest, but it
  leaves the paper without a method contribution.
- GraphCal (the proposed method) is high-risk and its win is modality-dependent.

It needs a thesis that is novel, general, supported by evidence we **already have**, and that
leaves a method artifact.

---

## 2. The finding the current framing under-uses (corrected against the data)

I first hypothesised "global test-time adaptation always beats per-input." **The data refutes
that** — and points to something better. From `calibration_bakeoff.json`:

| Modality | Per-input signal vs global S-norm | Verdict |
|---|---|---|
| **CZoo (chimp faces)** | quality-snorm 0.188 vs S-norm 0.227 @spatter-3; Δ=+3.3pt, **CI [0.012, 0.054]** | local **wins** (significant) |
| **IPanda50 (coat pattern)** | quality-snorm 0.370 vs S-norm 0.380; Δ=+1.0pt, **CI [−0.005, 0.023]** | **tie** (not significant) |

The same split appears on **three independent axes**:

| Axis (per-input structural signal) | Face-like modality | Repetitive-pattern modality |
|---|---|---|
| **Calibration** (quality-snorm vs S-norm) | CZoo: **wins +3.3pt** (sig) | IPanda: tie |
| **Re-ranking** (k-reciprocal → S-norm) | MacaqueFaces: **−2.5pt EER**, CZoo −0.5pt | IPanda: **worse** |
| **Fusion** (per-sample gate vs scalar) | — | Cattle muzzle (pattern): global scalar **wins**, gate worst |

**The consistent variable is not global-vs-local — it is the MODALITY.** Per-input *structural*
test-time signals help **face-like** modalities and are neutral/harmful for **repetitive-pattern**
modalities. The cattle-muzzle fusion result is simply the *pattern-side* confirmation of this law.
That convergence across calibration, re-ranking, and fusion — currently split across two papers as
separate negatives — is the strongest, most novel thing in the project.

> **Honesty note.** This corrects an earlier draft of this doc that claimed "global beats local."
> The bootstrap CIs show local *significantly beats* global on faces. The real law is
> modality-dependence, which is both truer and a better paper.

---

## 3. Candidate theses (ranked)

### ★ Recommended — "Structure-reliability governs test-time re-ID calibration"

**One line:** *Whether per-input structural test-time signals help cross-domain re-ID is
modality-dependent — significant gains on face-like modalities, neutral/harmful on
repetitive-pattern modalities — and this is predicted by a single measurable property of the
target embedding graph (reciprocal-neighbour purity). A label-free gate on that statistic gets
the best of both and is the first method to robustly match-or-beat S-norm across both.*

**Why it clears the bar:**
- **Novel + general.** A law about *when* structural test-time adaptation transfers, unifying
  three mechanisms (calibration, re-ranking, fusion) and two modality classes — not one trick.
- **Mechanistic + measurable.** The face-vs-pattern split becomes a computed statistic
  (reciprocal-neighbour purity / neighbourhood consistency) that **predicts the sign and size**
  of the structural-signal benefit *before* applying it. This is the make-or-break new result.
- **Method artifact.** A light **structure-reliability gate**: use structural calibration where
  the statistic is high (faces), fall back to S-norm where it is low (patterns). First thing to
  match-or-beat S-norm on **both** modality types — far more tractable than a meta-trained GNN.
- **Evidence mostly in hand.** The three-axis split is already measured (§2). New work = the
  statistic + its predictive validation + the gate.

**Experiments (mostly cheap):**
1. Formalise the split: one figure, per-input-structural benefit vs modality, across 5+ datasets
   × 2 backbones (MegaDescriptor, DINOv2). *(reuses existing runs)*
2. Define `reciprocal-purity(target)` and show it correlates with the measured benefit sign
   across datasets — **the core new result**. *(days)*
3. Analytic gate on the statistic; show ≥ S-norm everywhere, > S-norm on faces. *(days)*
4. Open-set × cross-domain headline: the gate recalibrates the *rejection threshold*.

**Risk:** low–medium. Even if the gate only *ties* S-norm on patterns, the law + predictive
statistic carry the paper. **Venue:** BMVC / WACV main track; analysis-friendly CVPR/ICCV track.

### Thesis B — GraphCal (learned meta-trained calibrator)
The learned generalisation of the gate: a GNN over the target similarity graph, meta-trained
across source domains, learning per-input when to trust structural signals. Highest ceiling,
high risk (meta-DG to unseen domains; must beat S-norm+k-reciprocal). **Treat as the scaled-up
version / future work of the recommended paper, not the whole bet.**

### Thesis A — Calibration–feature decomposition + deployment diagnostic
Split cross-domain error into an irrecoverable feature term and a label-free-recoverable
calibration term; an unsupervised statistic predicts the split (whether to collect target
labels). Very tractable; **fold in as the "how much is recoverable" quantifier.**

### Thesis E — Open-set × cross-domain calibration
The rejection threshold, not ranking, breaks under shift; label-free recalibration fixes the
field's hardest regime. Our domain paper already shows rejection is hardest. **Use as the
headline application of the recommended thesis.**

---

## 4. The recommended paper, assembled

**Title (draft):** *When Do Structural Test-Time Signals Help Cross-Domain Re-Identification?
A Modality Law and a Reliability Gate.*

**Arc:**
1. **Law (analysis).** Per-input structural test-time signals help faces, not patterns —
   shown on three axes (calibration, re-ranking, fusion), 5+ datasets, 2 backbones, with
   bootstrap CIs.
2. **Governing statistic (mechanism).** Reciprocal-neighbour purity predicts the benefit sign;
   validated across datasets. (Why: mutual-neighbour structure is informative for faces, noisy
   for repetitive coats/ridges.)
3. **Structure-reliability gate (method).** Minimal label-free module on the statistic; first to
   match-or-beat S-norm on both faces and patterns; baselines TENT / TDA / AdaBN / k-recip.
4. **Application (impact).** Open-set × cross-domain rejection-threshold recalibration.
5. **Future work.** GraphCal — learned, meta-trained generalisation.

**Why this beats the current draft:** it keeps everything proven (S-norm robustness, the
conditional law, modality-dependence) but reframes two negatives into one positive *law + statistic
+ method* — "here is exactly when and why to go beyond S-norm, and the minimal thing that does."

---

## 5. Immediate de-risking experiments (do first, fail fast)
1. **Formalise the modality split from existing data** — one figure across the three axes.
   Near-zero cost; if it isn't crisp, the thesis weakens.
2. **Define + test `reciprocal-purity`** on MacaqueFaces / CZoo / IPanda50 (+2 more) and
   correlate with measured benefit. **Make-or-break.**
3. **Analytic gate**; compare to S-norm on all datasets.
Only after (1)–(3) look good, commit to the full breadth sweep + baselines + open-set.

## 6. Honesty guardrails
- Bootstrap-test every "beats S-norm" claim (per-probe resampling) — as in §2, the CI is what
  separates the CZoo win from the IPanda tie.
- Verify suspicious wins in a second environment (the Modal-hybrid inflation lesson).
- The METHOD carries across species; the muzzle *modality* does not exist in wildlife.
- If the gate only ties S-norm on patterns, say so; the law + statistic still carry the paper.

## 7. BLOCKERS in the current calibration paper (from the paper review) — resolve before submission
These must be fixed during the rework; they are correctness/provenance issues, not style:
1. **Unbacked headline number (critical).** The DINOv2 / macaque-face / natural-shift numbers
   (29.2% → 20.7% EER, +8.4 pt, CI [+7.2,+9.8]) that anchor the abstract's "degrades by over
   twelve EER points" claim appear **only in the paper** — no committed JSON/log. `wildlife_natural_shift.py`
   supports `--backbone dinov2` but **saves nothing** (print-only). Action: re-run and **commit the
   raw output** (add `save_stats(...)` to the script), or remove the claim. Do not submit an
   unreproducible headline.
2. **Table 1 row 5 (MegaDescriptor/panda/natural) recovery vs CI.** Recovery listed **+0.3 pt**
   but its 95% CI is **[−0.33, +0.14]** (excludes +0.3), and rows 1/8 report the bootstrap-mean
   (not the point estimate). Report the bootstrap-mean (≈−0.04, n.s.) so it is consistent with its
   own CI and estimator.
3. **Abstract "over twelve EER points" vs body "29.2% EER"** — reconcile the anchor (degradation
   from DINOv2's own in-domain EER vs absolute) once (1) is regenerated.
4. **"large-EER regime" label** on MegaDescriptor rows (baseline EER 2.77%, 6.24%) sits awkwardly
   beside the panda-natural row (2.63%, n.s.). Since 2.63 ≈ 2.77, bootstrap-CI those rows or soften
   the label to stay consistent with the conditional law.
