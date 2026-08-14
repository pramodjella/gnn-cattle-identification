# Resume note — full-budget CV for CNN + ProtoN

**Status:** paused 2026-08-14 after fold 1 of 5 (stopped deliberately; ~12.5 h remained).
The paper ships without it — this is optional polish, not a mentor request.

## What is already done (in the paper)
- **Hybrid** full-budget 5-fold CV: **95.52 ± 0.31 Rank-1**, 0.49 ± 0.16 EER
  (`outputs/stats/cross_validation_hybrid_fullbudget.json`). This was the mentor's
  actual priority and it is complete.
- `Table 2` therefore mixes budgets: Hybrid = full, CNN/ProtoN = reduced. This is
  disclosed in the table caption and in limitation (i).

## What is pending
Full-budget CV for CNN (150 epochs) and ProtoN (200 epochs), to make Table 2 uniform.

### Fold 1 result (completed before the stop — do not lose this)
| Model | Full budget, fold 1 | Reduced-budget CV (in paper) | Delta |
|---|---|---|---|
| CNN | 97.34% R1, 0.05% EER | 93.91% | +3.4 pts |
| ProtoN | 94.38% R1, 1.00% EER | 89.49% | +4.9 pts |

Raw log preserved at `outputs/stats/cv_cnn_proton_fold1_partial.json`.
NOTE: one fold is not a CV result — do not put these in the paper as-is.

### Command to resume (runs all 5 folds from scratch)
```
venv/Scripts/python.exe scripts/cross_validation.py --models CNN ProtoN \
  --epochs-cnn 150 --epochs-proton 200 \
  --out cross_validation_cnn_proton_fullbudget.json
```
- Flag is `--out` (filename only, lands in `outputs/stats/`), **not** `--output`.
- Budget ≈ **3 h 07 m per fold** (CNN 2 h 39 m + ProtoN 27 m) → **~15.5 h for 5 folds**.
  Start it in the morning or run overnight.
- Runs locally on purpose: Modal is untrusted for these models (environment-inflated
  results — see the modal-ablation findings).

### When it finishes
1. Update `Table 2` (`tab:cv`) in `docs/paper_domain_arxiv.tex` **and** the matching
   table in `paper_draft.md` with the new CNN/ProtoN rows; mark all three rows `full`.
2. Delete limitation **(i)** (it only exists because those two rows are reduced-budget)
   and renumber the remaining limitations.
3. Keep the §3.2 paragraph explaining why CV > single-split (80% vs 70% training data;
   folds evaluate the final model, the single split picks a best-val checkpoint on a
   noisy 2.4-img/identity partition). It applies to all three models.
4. Recompile, rebuild deliverables, commit, push:
   - `tectonic -X compile docs/paper_domain_arxiv.tex --outdir .` (binary in scratchpad)
   - `NODE_PATH=<scratchpad>/docxbuild/node_modules node docs/_build_paper_docx.js`
   - refresh the three files in `~/Downloads` (PDF, DOCX, arXiv zip)
