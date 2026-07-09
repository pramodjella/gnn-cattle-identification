# Submission Runbook — IJCB / WACV / BMVC

Target: a competitive biometrics/CV conference. This file sequences the
experiments that move the paper from "sound" to "competitive". Items marked
**[DONE]** are implemented and verified; **[RUN]** items need your GPU.

---

## 0. Already done this session (no GPU needed)

- **[DONE]** Leakage/protocol audit — `python scripts/verify_data_integrity.py`
  (closed-set verified, 0 image leaks; results in `outputs/stats/data_integrity.json`).
- **[DONE]** Honest validation-selected ensemble — `python scripts/ensemble_inference.py`
  (test Rank-1 96.1%, EER 0.78%, selection gap 0.00).
- **[DONE]** Quantitative explainability — `python scripts/evaluate_explainability.py --model gnn_v3`
  (Fidelity+/-, sparsity, cross-method agreement).
- **[DONE]** Open-set evaluation — `python scripts/evaluate_openset.py --model cnn --known-frac 0.6`
  (Rank-1 known 99.1%, open-set AUC 0.984, DIR@FAR=1% 61.2%).
- **[DONE]** New architecture options: multi-scale node sampling + adaptive
  learned-graph construction (`config.yaml` → `hybrid.multi_scale`,
  `hybrid.learned_edges`). Backward compatible with existing checkpoints.

---

## 1. Make the GNN earn its place  **[RUN]**

Retrain the Hybrid with the new components at full budget. Toggle in `config.yaml`:

```yaml
hybrid:
  multi_scale: true         # FPN-style 96+232+1536-ch node features
  learned_edges: true       # adaptive graph construction
  edge_prune_threshold: 0.1
```

Then:
```powershell
venv\Scripts\python.exe scripts/train_hybrid.py
venv\Scripts\python.exe scripts/ensemble_inference.py     # re-tune blend on val
```

**Success criterion:** Hybrid single-model Rank-1 ≥ CNN (95.4%), OR the blended
EER drops further. If Rank-1 still trails, keep the honest "graphs improve
verification" framing — do not overclaim.

**Ablation to add (required for the novelty claim):**
| Variant | multi_scale | learned_edges |
| :-- | :--: | :--: |
| Hybrid (base) | off | off |
| + multi-scale | on | off |
| + adaptive graph | off | on |
| + both (full) | on | on |

Run each ~2–3 short seeds; report mean ± std Rank-1 and EER. This isolates the
contribution of each named component.

---

## 2. Full-budget 5-fold CV of the Hybrid  **[RUN]**  (mandatory)

Table 2 currently under-reports the Hybrid (12-epoch reduced budget). Re-run at
full budget so it reconciles with the single-split 92%:

```powershell
venv\Scripts\python.exe scripts/cross_validation.py --epochs-cnn 100 --epochs-proton 150 --epochs-hybrid-p1 150 --epochs-hybrid-p2 40
```
Report Rank-1/EER mean ± std for CNN, ProtoN, and Hybrid on equal footing.

---

## 3. Open-set with held-out identities  **[RUN]**  (strong add for IJCB)

The current open-set script evaluates a closed-set-trained model. For a strict
"generalisation to unseen animals" claim, retrain on a subset of identities and
test on the held-out ones:

1. Create an identity-disjoint split (e.g. train on 180 IDs, evaluate open-set
   on the remaining 80 as unknown probes).
2. Retrain CNN (+ Hybrid) on the 180-ID training set.
3. `python scripts/evaluate_openset.py` with the held-out gallery/probe.

Report DIR@FAR and open-set ROC across ≥3 known/unknown partitions (seeds).

---

## 4. Second dataset / cross-dataset  **[RUN]**  (biggest top-tier lift)

**Chosen second dataset:** Pakistan "Cows Frontal Face / Muzzle" set — Zenodo
record `10535934` (DOI 10.5281/zenodo.8377921), 459 individuals, ~2,893 images,
CC BY 4.0. Different country/breeds/cameras than the US-feedlot training set.
Frontal-face images with muzzle YOLO boxes -> crop the muzzle first.

```powershell
# download (~13.9 GB) + extract
venv\Scripts\python.exe scripts/prepare_cross_dataset.py --zenodo-record 10535934 --download --extract
# organize to folder-per-animal, cropping the muzzle box if .txt labels exist
venv\Scripts\python.exe scripts/prepare_cross_dataset.py --organize --crop-muzzle `
    --src data/external/10535934/extracted --out data/external/pakistan_muzzle
# zero-shot transfer eval (no fine-tuning)
venv\Scripts\python.exe scripts/evaluate_cross_dataset.py --data-root data/external/pakistan_muzzle
```
The preparer auto-finds YOLO labels next to images OR in a sibling `labels/`
dir (standard `images/`<->`labels/` split). If NO labels exist and the images
are full faces, the center/full image is a poor muzzle ROI — inspect the
archive or run a detector first (the tool warns you when this happens).

**Kaggle alternatives** (token auth — never a password): create an API token at
Kaggle → Account → *Create New API Token*, save `kaggle.json` to
`%USERPROFILE%\.kaggle\`, then:
```powershell
venv\Scripts\pip install kaggle
venv\Scripts\python.exe scripts/prepare_cross_dataset.py --kaggle-dataset kollabathulakaushik/cattle-images-db-for-muzzle-based-identification --download --organize --out data/external/kaggle_muzzle
venv\Scripts\python.exe scripts/evaluate_cross_dataset.py --data-root data/external/kaggle_muzzle
```
Candidates: `kollabathulakaushik/cattle-images-db-for-muzzle-based-identification`,
`sharifashik/cow-muzzle-dataset`. **Indian option:** the Vrindavani crossbred
set (264 animals, 2,640 imgs) is paywalled — request from the author
(ayontarafdar@gmail.com) to add an Indian dataset if time allows.

## 5. Modal cloud ablation  **[RUN]**  (fast full matrix)

The local RTX 5070 (8 GB) handles single runs and all evals; use Modal for the
4-variant x CV matrix. One-time data upload then:
```powershell
modal run modal_ablation.py
modal volume get cattle-gnn-data /results ./outputs/stats_modal
```
See header of `modal_ablation.py` for the exact `modal volume put` upload
commands. Then aggregate:  `python scripts/run_all_experiments.py --aggregate`.

---

## 6. Paper polish  **[RUN cheap]**

- Regenerate figures: `python scripts/figures/generate_paper_figures.py`
- Add faithfulness table + open-set table + ablation table to `paper_draft.md`.
- Convert `paper_draft.md` to the venue LaTeX template; move contributions to
  match the venue's expected structure.

---

## Acceptance-risk checklist

| Risk a reviewer raises | Status |
| :-- | :-- |
| Leakage (test > val) | Resolved + audited (§2.1) |
| Ensemble tuned on test | Resolved (val-selected, gap 0.00) |
| CV contradicts test table | Explained; full-budget CV pending (**§2**) |
| GNN loses to CNN | Reframed as verification win; **§1** may flip it |
| Thin novelty | Adaptive graph construction is the named contribution (**§1**) |
| No open-set | Implemented; held-out-ID version pending (**§3**) |
| Single dataset | Open item (**§4**) |
| Qualitative-only explainability | Resolved (faithfulness metrics) |
