# run_tuned_pipeline.ps1
# TUNED Pipeline for 98%+ Accuracy — Sequential Execution
# -------------------------------------------------------------------------
# Runs all tuned model training in sequence to prevent Windows paging exhaustion.
# Based on research findings and hyperparameter tuning for 98%+ Rank-1.
#
# TUNING SUMMARY:
#   CNN:    EfficientNet-B4 + Mixup + SWA + TTA | scale=128, margin=0.35, emb=512
#   Hybrid: 200 epoch cached + 50 epoch E2E | EnhancedGraphAug | scale=96, m=0.35
#   ProtoN: align_weight=0.2 (was 0.5) | 4 layers | dropout=0.12
#
# Sequence:
#   1. Clear stale Hybrid feature cache (B3 -> B4 upgrade)
#   2. Train CNN      (EfficientNet-B4 + ArcFace + Mixup + SWA)    bs=16  ~3-4 hrs
#   3. Train GNN v3   (GATv2, 4L, hidden=192, fusion=768)          bs=64  ~4-5 hrs
#   4. Train GNN v4   (GATv2, 4L, 8 heads, VRAM-auto)              bs=32  ~5-6 hrs
#   5. Train ProtoN   (align_weight=0.2, 4 layers, dropout=0.12)   bs=128 ~4-5 hrs
#   6. Train Hybrid   (EnhancedAug, 200+50ep E2E finetune)         bs=32  ~3-4 hrs
#   7. TTA evaluation on all models
#   8. Ensemble inference (CNN + Hybrid)
#   9. Final comparison report
#
# VRAM budget (RTX 5070 8GB):
#   CNN B4:    ~5.5GB @ bs=16   (safe)
#   GNN v3:    ~4.5GB @ bs=64   (wider 768-d head)
#   GNN v4:    ~6.0GB @ bs=32   (1024-d head, 8 heads)
#   ProtoN:    ~3.0GB @ bs=128  (512-d head, 4 heads)
#   Hybrid:    ~4.0GB @ bs=32   (GNN portion only in phase 1)
# -------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$PythonExe = "F:\GNN Research\gnn-cattle-identification\venv\Scripts\python.exe"
$ProjectRoot = "F:\GNN Research\gnn-cattle-identification"
$env:PYTHONIOENCODING = "utf-8"

function Run-Step {
    param([string]$StepName, [string]$Script)
    Write-Host "`n$('='*65)" -ForegroundColor Cyan
    Write-Host "  $StepName" -ForegroundColor Cyan
    Write-Host "$('='*65)" -ForegroundColor Cyan
    $startTime = Get-Date
    & $PythonExe $Script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $StepName failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $elapsed = (Get-Date) - $startTime
    Write-Host "  [OK] $StepName completed in $([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s" -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TUNED TRAINING PIPELINE -- Target: 98%+ Rank-1"           -ForegroundColor Cyan
Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"       -ForegroundColor Cyan
Write-Host "  VRAM Budget (8GB RTX 5070):"                              -ForegroundColor Cyan
Write-Host "    CNN B4:  bs=16  -> ~5.5GB | GNN v3: bs=64  -> ~4.5GB"  -ForegroundColor Cyan
Write-Host "    GNN v4:  bs=32  -> ~6.0GB | ProtoN: bs=128 -> ~3.0GB"  -ForegroundColor Cyan
Write-Host "    Hybrid:  bs=32  -> ~4.0GB"                              -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

Set-Location "F:\GNN Research\gnn-cattle-identification"

# ── Step 1: Clear stale Hybrid cache (B3→B4 upgrade means cache is invalid) ──
Write-Host "`n[Step 1] Clearing stale Hybrid feature cache (B3 → B4 upgrade)..." -ForegroundColor Yellow
$cacheDir = "outputs\hybrid\feature_cache"
if (Test-Path $cacheDir) {
    Remove-Item -Recurse -Force $cacheDir
    Write-Host "  -> Deleted stale cache: $cacheDir" -ForegroundColor Green
} else {
    Write-Host "  -> No stale cache found (clean start)" -ForegroundColor Green
}

# ── Step 2: Train CNN (TUNED) ─────────────────────────────────────────────────
Run-Step "CNN Training [B4+ArcFace+Mixup+SWA | bs=16 | ~3-4hrs]" "scripts\train_cnn.py"

# ── Step 3: Train GNN v3 (TUNED) ──────────────────────────────────────────────
Run-Step "GNN v3 Training [GATv2 4L hidden=192 | bs=64 | ~4-5hrs]" "scripts\train_gnn_v3_optimized.py"

# ── Step 4: Train GNN v4 (TUNED) ──────────────────────────────────────────────
Run-Step "GNN v4 Training [GATv2 4L 8heads | bs=32 VRAM-auto | ~5-6hrs]" "scripts\train_gnn_v4_enhanced.py"

# ── Step 5: Train ProtoN (TUNED) ──────────────────────────────────────────────
Run-Step "ProtoN Training [align=0.2, 4L, dropout=0.12 | bs=128 | ~4-5hrs]" "scripts\train_proton.py"

# ── Step 6: Train Hybrid (TUNED) ──────────────────────────────────────────────
Run-Step "Hybrid CNN-GNN Training [EnhancedAug+50ep E2E | bs=32 | ~3-4hrs]" "scripts\train_hybrid.py"

# ── Step 7: TTA Evaluation ───────────────────────────────────────────────────
Run-Step "TTA Evaluation -- All Models" "scripts\evaluate_with_tta.py"

# ── Step 8: Ensemble Inference ────────────────────────────────────────────────
Run-Step "Ensemble Inference (CNN + Hybrid)" "scripts\ensemble_inference.py"

# ── Step 9: Final Comparison ──────────────────────────────────────────────────
$compareScript = "scripts\compare_models.py"
if (Test-Path $compareScript) {
    Run-Step "Final Model Comparison Report" $compareScript
} else {
    Write-Host "`n[Step 9] compare_models.py not found -- skipping" -ForegroundColor Yellow
}

Write-Host "`n$('='*65)" -ForegroundColor Green
Write-Host "  TUNED PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "  Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "  Check outputs/stats/ for all results" -ForegroundColor Green
Write-Host "  Key files:" -ForegroundColor Green
Write-Host "    cnn_results.json          -- CNN B4 + Mixup + SWA + TTA" -ForegroundColor Green
Write-Host "    gnn_v3_optimized_results.json -- GNN v3 (dropout=0.10, wider)" -ForegroundColor Green
Write-Host "    gnn_v4_enhanced_results.json  -- GNN v4 (dropout=0.15, wd=1e-5)" -ForegroundColor Green
Write-Host "    proton_results.json        -- ProtoN (align_weight=0.2)" -ForegroundColor Green
Write-Host "    hybrid_results.json        -- Hybrid 50-ep E2E finetune" -ForegroundColor Green
Write-Host "    ensemble_results.json      -- Ensemble (CNN+Hybrid)" -ForegroundColor Green
Write-Host "    tta_evaluation_summary.json -- TTA gains per model" -ForegroundColor Green
Write-Host "$('='*65)" -ForegroundColor Green
