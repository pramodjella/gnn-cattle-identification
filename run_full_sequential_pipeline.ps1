# run_full_sequential_pipeline.ps1
# Complete Cattle Identification Pipeline (Sequential Execution)
# -------------------------------------------------------------------------
# This script executes the entire pipeline sequentially to prevent
# Windows paging file exhaustion (Error 1455) caused by parallel data loading.
#
# Kornia backends available (pass --backend to step 1):
#   disk        – Kornia DISK  (default, recommended)
#   superpoint  – Kornia KeyNet + AffNet + HardNet8
#   dedode      – Kornia DeDoDe
#   sift        – OpenCV SIFT  (classical baseline)
#
# Sequence:
# 1. Back up original SIFT graphs
# 2. Generate new Kornia graphs (skip if already done; use --force to redo)
# 3. Train GNN+ (Kornia Baseline)
# 4. Extract MobileNetV3 Patch Features (for GNN++)
# 5. Train GNN++ (MobileNetV3 + Kornia)
# 6. Train CNN (EfficientNet-B3 + ArcFace)
# 7. Train Hybrid (CNN-GNN)
# 8. Evaluate Hybrid (generate results JSON if checkpoint exists)
# 9. Multi-method Keypoint Benchmark (DISK vs SuperPoint vs DeDoDe vs SIFT)
# 10. Generate final paper comparisons
# 11. Generate Dual Explainability Heatmaps
# -------------------------------------------------------------------------

# ── CLI Params ────────────────────────────────────────────────────────────────
param(
    [string]$Backend   = "disk",   # disk | superpoint | dedode | sift
    [switch]$ForceKP,              # force keypoint re-extraction
    [switch]$SkipTrain             # skip training steps (benchmark/eval only)
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  STARTING FULL SEQUENTIAL PIPELINE (Kornia Edition)"      -ForegroundColor Cyan
Write-Host "  Backend: $Backend  ForceKP: $ForceKP  SkipTrain: $SkipTrain" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 0: Backup SIFT graphs
Write-Host "`n[Step 0] Backing up original SIFT graphs..." -ForegroundColor Yellow
if (-not (Test-Path "data/graphs_sift")) {
    New-Item -ItemType Directory -Force -Path "data/graphs_sift" | Out-Null
    $siftFiles = Get-ChildItem -Path "data/graphs" -Filter "*_graphs.pt" -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -notlike "*_v2*" }
    if ($siftFiles) {
        foreach ($f in $siftFiles) {
            $dest = "data/graphs_sift/$($f.Name)"
            if (-not (Test-Path $dest)) {
                Copy-Item -Path $f.FullName -Destination $dest -Force
            }
        }
        Write-Host "  -> Original SIFT graphs backed up to data/graphs_sift/" -ForegroundColor Green
    } else {
        Write-Host "  -> No original graphs found to backup." -ForegroundColor Yellow
    }
} else {
    Write-Host "  -> Backup directory already exists. Skipping." -ForegroundColor Green
}

# Step 1: Generate Kornia Graphs
$kpFlag  = if ($ForceKP) { "--force" } else { "" }
$kpArgs  = @("scripts\03_extract_keypoints.py", "--backend", $Backend)
if ($ForceKP) { $kpArgs += "--force" }

$trainGraphs = "data/graphs/train_graphs.pt"
if ((Test-Path $trainGraphs) -and (-not $ForceKP)) {
    Write-Host "`n[Step 1] Kornia graphs already exist. Skipping extraction." -ForegroundColor Green
    Write-Host "         (Use -ForceKP to re-extract with backend: $Backend)" -ForegroundColor DarkGray
} else {
    Write-Host "`n[Step 1] Extracting keypoints with backend: $Backend ..." -ForegroundColor Yellow
    & .\venv\Scripts\python.exe @kpArgs
    if ($LASTEXITCODE -ne 0) { throw "Keypoint extraction failed" }
    Write-Host "`n[Step 1b] Building graphs..." -ForegroundColor Yellow
    .\venv\Scripts\python.exe scripts\04_build_graphs.py
    if ($LASTEXITCODE -ne 0) { throw "Graph builder failed" }
}

if (-not $SkipTrain) {

    # Step 2: Train GNN+ (Kornia Baseline)
    $gnnPlusResult = "outputs/stats/gnn_plus_results.json"
    if ((Test-Path $gnnPlusResult) -and (Get-Item $gnnPlusResult).Length -gt 1000) {
        Write-Host "`n[Step 2] GNN+ results already exist. Skipping training." -ForegroundColor Green
    } else {
        Write-Host "`n[Step 2] Training GNN+ (Kornia Baseline)..." -ForegroundColor Yellow
        .\venv\Scripts\python.exe scripts\train_gnn_plus.py
        if ($LASTEXITCODE -ne 0) { throw "GNN+ training failed" }
    }

    # Step 3: Extract GNN++ MobileNetV3 Patch Features
    $v2Train = "data/graphs/train_graphs_v2.pt"
    if (Test-Path $v2Train) {
        Write-Host "`n[Step 3] GNN++ v2 graphs already exist. Skipping extraction." -ForegroundColor Green
    } else {
        Write-Host "`n[Step 3] Extracting MobileNetV3 Patch Features for GNN++..." -ForegroundColor Yellow
        .\venv\Scripts\python.exe scripts\extract_patch_features.py
        if ($LASTEXITCODE -ne 0) { throw "Patch extraction failed" }
    }

    # Step 4: Train GNN++ (MobileNetV3 + Kornia features)
    $gnnPlusPlusResult = "outputs/stats/gnn_plus_v2_results.json"
    if ((Test-Path $gnnPlusPlusResult) -and (Get-Item $gnnPlusPlusResult).Length -gt 1000) {
        Write-Host "`n[Step 4] GNN++ results already exist. Skipping training." -ForegroundColor Green
    } else {
        Write-Host "`n[Step 4] Training GNN++ (MobileNetV3 Patches + 4-layer ResEdgeConv)..." -ForegroundColor Yellow
        .\venv\Scripts\python.exe scripts\train_gnn_plus_v2.py
        if ($LASTEXITCODE -ne 0) { throw "GNN++ training failed" }
    }

    # Step 5: Train CNN (EfficientNet-B3)
    $cnnResult = "outputs/stats/cnn_results.json"
    if ((Test-Path $cnnResult) -and (Get-Item $cnnResult).Length -gt 1000) {
        Write-Host "`n[Step 5] CNN results already exist. Skipping training." -ForegroundColor Green
    } else {
        Write-Host "`n[Step 5] Training CNN Baseline (EfficientNet-B3)..." -ForegroundColor Yellow
        .\venv\Scripts\python.exe scripts\train_cnn.py
        if ($LASTEXITCODE -ne 0) { throw "CNN training failed" }
    }

    # Step 6: Train Hybrid CNN-GNN
    $hybridResult = "outputs/stats/hybrid_results.json"
    $hybridCkpt   = "outputs/hybrid/best_model.pt"
    if ((Test-Path $hybridResult) -and (Get-Item $hybridResult).Length -gt 1000) {
        Write-Host "`n[Step 6] Hybrid results already exist. Skipping training." -ForegroundColor Green
    } elseif (Test-Path $hybridCkpt) {
        Write-Host "`n[Step 6] Hybrid checkpoint exists. Running evaluation only..." -ForegroundColor Yellow
        .\venv\Scripts\python.exe scripts\eval_hybrid.py
        if ($LASTEXITCODE -ne 0) { throw "Hybrid evaluation failed" }
    } else {
        Write-Host "`n[Step 6] Training Hybrid CNN-GNN..." -ForegroundColor Yellow
        .\venv\Scripts\python.exe scripts\train_hybrid.py
        if ($LASTEXITCODE -ne 0) { throw "Hybrid training failed" }
    }

} else {
    Write-Host "`n[Steps 2-6] SkipTrain flag set. Skipping all training." -ForegroundColor DarkGray
}

# Step 7: Re-evaluate all models (fill in any missing JSON fields)
Write-Host "`n[Step 7] Re-evaluating models to fill missing result fields..." -ForegroundColor Yellow
.\venv\Scripts\python.exe scripts\reeval_all_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Re-evaluation had issues. Continuing..." -ForegroundColor Yellow
}

# Step 8: Multi-method Keypoint Benchmark (parallel: DISK, SuperPoint, DeDoDe, SIFT)
Write-Host "`n[Step 8] Running parallel multi-method keypoint benchmark..." -ForegroundColor Yellow
Write-Host "         (Kornia DISK | SuperPoint/KeyNet+HardNet | DeDoDe | SIFT)" -ForegroundColor DarkGray
.\venv\Scripts\python.exe scripts\compare_sift_vs_kornia.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Keypoint benchmark failed (non-critical). Continuing..." -ForegroundColor Yellow
}

# Step 9: Compare Models and Generate Report
Write-Host "`n[Step 9] Generating Final Paper Results and Comparisons..." -ForegroundColor Yellow
.\venv\Scripts\python.exe scripts\compare_models.py
if ($LASTEXITCODE -ne 0) { throw "Comparison script failed" }

# Step 10: Generate Dual Explainability Heatmaps
Write-Host "`n[Step 10] Generating Hybrid Dual-Explainability Heatmaps..." -ForegroundColor Yellow
.\venv\Scripts\python.exe scripts\visualize_attention.py --model hybrid --num_samples 5
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Visualization failed (non-critical). Continuing..." -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  FULL PIPELINE COMPLETED SUCCESSFULLY"                      -ForegroundColor Green
Write-Host "  Results saved in:"                                          -ForegroundColor Green
Write-Host "    outputs/stats/     - JSON metrics for all models"         -ForegroundColor Green
Write-Host "    outputs/figures/   - CMC/ROC curves + multi-method plots" -ForegroundColor Green
Write-Host "    outputs/results/   - publication_report.md"              -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Usage examples:" -ForegroundColor Cyan
Write-Host "    .\run_full_sequential_pipeline.ps1                        (default DISK)" -ForegroundColor DarkGray
Write-Host "    .\run_full_sequential_pipeline.ps1 -Backend superpoint    (KeyNet+HardNet)" -ForegroundColor DarkGray
Write-Host "    .\run_full_sequential_pipeline.ps1 -Backend disk -ForceKP (force re-extract)" -ForegroundColor DarkGray
Write-Host "    .\run_full_sequential_pipeline.ps1 -SkipTrain             (benchmark only)" -ForegroundColor DarkGray
