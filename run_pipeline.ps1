# ============================================================
# GNN Cattle Identification - Full Pipeline Runner
# Target: NVIDIA RTX 5070 (8GB VRAM) | CUDA 12.8
# ============================================================

$ErrorActionPreference = "Stop"
$VENV_PYTHON = ".\venv\Scripts\python.exe"
$SCRIPTS = ".\scripts"

function Run-Step {
    param([string]$Script, [string]$Name)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  RUNNING: $Name" -ForegroundColor Yellow
    Write-Host ("=" * 70) -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $VENV_PYTHON "$SCRIPTS\$Script"
    $sw.Stop()
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAILED] $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ""
    Write-Host "[OK] $Name completed in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) min" -ForegroundColor Green
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Magenta
Write-Host "  GNN CATTLE IDENTIFICATION - FULL PIPELINE" -ForegroundColor Magenta
Write-Host "  Device: NVIDIA RTX 5070 (8GB VRAM) | CUDA 12.8" -ForegroundColor Magenta
Write-Host ("=" * 70) -ForegroundColor Magenta

# Check GPU
& $VENV_PYTHON -c "import torch; print(f'[GPU] {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM | CUDA {torch.version.cuda}')"

$global_sw = [System.Diagnostics.Stopwatch]::StartNew()

Run-Step "01_download_data.py"   "Phase 1: Download & Prepare Dataset"
Run-Step "02_preprocess.py"      "Phase 2: Image Preprocessing (CLAHE + Segmentation)"
Run-Step "03_extract_keypoints.py" "Phase 3: Keypoint Extraction (SuperPoint)"
Run-Step "04_build_graphs.py"    "Phase 4: Graph Construction (KNN)"
Run-Step "05_train.py"           "Phase 5: Train CattleGNN on RTX 5070"
Run-Step "06_evaluate.py"        "Phase 6: Evaluate Model"
Run-Step "07_generate_paper_stats.py" "Phase 7: Generate Paper Statistics"

$global_sw.Stop()
$total = [math]::Round($global_sw.Elapsed.TotalMinutes, 2)

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Magenta
Write-Host "  PIPELINE COMPLETE in $total minutes" -ForegroundColor Green
Write-Host "  Results in: outputs/" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Magenta
