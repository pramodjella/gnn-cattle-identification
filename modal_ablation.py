"""
Modal Cloud Ablation Harness
============================
Runs the 4-variant Hybrid ablation (and optional CV) on Modal GPUs in parallel,
so the full matrix finishes in wall-clock hours instead of GPU-days on a local
8 GB card.

One-time data upload (from your machine, after `modal token new`):
    modal volume create cattle-gnn-data
    modal volume put cattle-gnn-data ./data/preprocessed   /data/preprocessed
    modal volume put cattle-gnn-data ./data/graphs         /data/graphs
    modal volume put cattle-gnn-data ./outputs/cnn         /outputs/cnn        # for ensembles
    # (graphs + preprocessed images are the inputs train_hybrid.py needs)

Run the ablation:
    modal run modal_ablation.py

Fetch results:
    modal volume get cattle-gnn-data /outputs/stats ./outputs/stats_modal
    python scripts/run_all_experiments.py --aggregate

Notes
-----
* GPU default is A10G (24 GB) — comfortably fits multi-scale + learned edges at
  a larger batch than the local 5070. Bump to "a100" for the fastest runs.
* This script is provided ready-to-run but has NOT been executed here (no Modal
  credentials in this session); verify the volume paths match your upload.
"""

import modal

app = modal.App("cattle-gnn-ablation")

# ── Container image: project deps + the repo source ──────────────────────────
# Add ONLY code dirs (validated) — never the multi-GB data/ or outputs/ dirs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")           # opencv runtime deps
    # PINNED to the local env so the hybrid's DynamicEdgeConv/scatter graph
    # forward is identical to what produced the paper numbers. torch-geometric
    # 2.8 (+ torch-cluster) computes the kNN/pooling differently and inflated
    # the hybrid eval by ~11 pts; 2.7 with NO torch-scatter/cluster matches local.
    .pip_install(
        "torch==2.11.0", "torchvision==0.26.0",
        "torch-geometric==2.7.0",
        "opencv-python-headless>=4.8.0", "kornia>=0.7.0",
        "scikit-image>=0.21.0", "scikit-learn>=1.3.0", "scipy>=1.11.0",
        "numpy>=1.24.0", "pandas>=2.0.0", "Pillow>=10.0.0",
        "PyYAML>=6.0", "omegaconf>=2.3.0", "matplotlib>=3.7.0", "tqdm>=4.65.0",
        "seaborn>=0.12.0", "tensorboard>=2.14.0",
    )  # deliberately NO torch-scatter / torch-cluster (match local fallback)
    .add_local_dir("src", "/root/project/src")
    .add_local_dir("scripts", "/root/project/scripts")
    .add_local_dir("config", "/root/project/config")
)

# Epoch budget for the ablation (relative comparison; early-stopping still applies).
# Override with MODAL_HYBRID_P1 / MODAL_HYBRID_P2 env vars for the full budget.
import os as _os
HYBRID_P1 = int(_os.environ.get("MODAL_HYBRID_P1", "80"))
HYBRID_P2 = int(_os.environ.get("MODAL_HYBRID_P2", "20"))

data_vol = modal.Volume.from_name("cattle-gnn-data", create_if_missing=True)

# Variant matrix: (name, multi_scale, learned_edges)
VARIANTS = [
    ("hybrid_base",       False, False),
    ("hybrid_multiscale", True,  False),
    ("hybrid_adaptive",   False, True),
    ("hybrid_full",       True,  True),
]


@app.function(image=image, gpu="a10g", volumes={"/data": data_vol},
              timeout=6 * 60 * 60)
def train_variant(name: str, multi_scale: bool, learned_edges: bool,
                  epochs_p1: int = 80, epochs_p2: int = 20) -> dict:
    """Patch config for one variant, train + eval the Hybrid, return metrics.

    Epoch budget is passed as explicit args (Modal serialises these), NOT via
    module globals — the container re-reads env vars, which would be unset.
    """
    import os, sys, json, shutil, subprocess
    import yaml

    proj = "/root/project"
    os.chdir(proj)
    sys.path.insert(0, proj)

    # Symlink shared read-only inputs (data) and a PER-VARIANT outputs tree on
    # the volume (parallel containers must not share the outputs/stats dir).
    for sub in ("preprocessed", "graphs"):
        src, dst = f"/data/{sub}", f"{proj}/data/{sub}"
        os.makedirs(f"{proj}/data", exist_ok=True)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    variant_out = f"/data/outputs_{name}"
    os.makedirs(f"{variant_out}/stats", exist_ok=True)
    if not os.path.exists(f"{proj}/outputs"):
        os.symlink(variant_out, f"{proj}/outputs")
    os.makedirs("/data/results", exist_ok=True)   # shared collection dir

    # Patch config.yaml -> hybrid variant flags + a distinct checkpoint dir.
    cfg_path = f"{proj}/config/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("hybrid", {})
    cfg["hybrid"]["multi_scale"] = multi_scale
    cfg["hybrid"]["learned_edges"] = learned_edges
    cfg["hybrid"]["checkpoint_dir"] = f"outputs/hybrid_{name}"
    cfg["hybrid"]["epochs"] = epochs_p1
    cfg["hybrid"]["finetune_epochs"] = epochs_p2
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    env = {**os.environ, "PYTHONPATH": proj}
    subprocess.run([sys.executable, "scripts/train_hybrid.py"], check=True, env=env)
    subprocess.run([sys.executable, "scripts/eval_hybrid.py"], check=False, env=env)

    # Collect the per-variant results written by eval_hybrid.py.
    res_path = f"{proj}/outputs/stats/hybrid_results.json"
    metrics = {}
    if os.path.exists(res_path):
        with open(res_path) as f:
            metrics = json.load(f)
        shutil.copy(res_path, f"/data/results/hybrid_{name}_results.json")

    data_vol.commit()
    return {"variant": name, "multi_scale": multi_scale,
            "learned_edges": learned_edges, "metrics": metrics}


@app.local_entrypoint()
def smoke():
    """Cheap end-to-end validation: one variant, tiny budget."""
    import json
    r = train_variant.remote("hybrid_base", False, False, 2, 1)
    print("\n===== SMOKE RESULT =====")
    print(json.dumps(r, indent=2, default=str))


@app.local_entrypoint()
def main():
    import json
    p1, p2 = HYBRID_P1, HYBRID_P2   # read locally (env is set here)
    # Optional filter: MODAL_VARIANTS="hybrid_multiscale,hybrid_full" runs a subset.
    only = _os.environ.get("MODAL_VARIANTS", "").strip()
    selected = VARIANTS
    if only:
        keep = {s.strip() for s in only.split(",")}
        selected = [v for v in VARIANTS if v[0] in keep]
    args = [(n, ms, le, p1, p2) for (n, ms, le) in selected]
    print(f"Launching {len(args)} variants at budget P1={p1}, P2={p2}")
    # return_exceptions: one variant failing must not cancel the others.
    results = list(train_variant.starmap(args, return_exceptions=True))
    summary = {}
    for (n, *_), r in zip(args, results):
        if isinstance(r, Exception):
            print(f"  [FAILED] {n}: {type(r).__name__}: {r}")
            summary[n] = {"variant": n, "error": str(r)}
        else:
            summary[n] = r
    print("\n===== ABLATION RESULTS =====")
    for name, r in summary.items():
        m = r.get("metrics", {})
        r1 = m.get("test_rank1") or m.get("rank1")
        eer = m.get("eer")
        print(f"  {name:20s} Rank-1={r1}  EER={eer}")
    with open("outputs/stats/ablation_hybrid_variants.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved -> outputs/stats/ablation_hybrid_variants.json")
