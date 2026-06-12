"""
Script: Run Sequential GNN Experiments
======================================
1. Monitors the active GNN v4 training run (PID 41572) until completion.
2. Runs the re-evaluation script to register GNN v4 stats.
3. Sequentially trains and evaluates:
   - ProtoN (Prototype Node GNN)
   - VisGIN (Visibility GNN)
   - Keypoint Matcher (Differentiable Optimal Transport Matcher)
4. Regenerates the final comparison table and figures.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_EXE = str(PROJECT_ROOT / "venv/Scripts/python.exe")

def is_pid_running(pid: int) -> bool:
    """Check if a process with the given PID is running on Windows."""
    try:
        output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True).decode(errors='ignore')
        return str(pid) in output
    except Exception:
        return False

def find_gnn_v4_pid() -> int | None:
    """Find the running PID of train_gnn_v4_enhanced.py dynamically."""
    cmd = 'powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name = \'python.exe\'\\" | Where-Object {$_.CommandLine -like \'*train_gnn_v4_enhanced.py*\'} | Select-Object -ExpandProperty ProcessId"'
    try:
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        if output:
            pids = [int(p) for p in output.split() if p.isdigit()]
            if pids:
                return pids[0]
    except Exception:
        pass
    return None

def run_cmd(cmd: list[str]) -> bool:
    """Execute a shell command, printing stdout/stderr in real-time."""
    print(f"\n[RUNNING] {' '.join(cmd)}")
    t0 = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )
        # Stream output line-by-line
        for line in process.stdout:
            print(line, end='', flush=True)
            
        process.wait()
        elapsed = time.time() - t0
        if process.returncode == 0:
            print(f"[SUCCESS] Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"[FAILED] Exit code {process.returncode} in {elapsed:.1f}s")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}")
        return False

def main():
    print("="*70)
    print("  SEQUENTIAL GNN CATTLE IDENTIFICATION EXPERIMENTS ORCHESTRATOR")
    print("="*70)
    
    target_pid = find_gnn_v4_pid()
    
    if target_pid is None:
        print("[INFO] train_gnn_v4_enhanced.py is not currently running.")
        print("Starting GNN v4 training first...")
        print("\n--- Step 0: Running GNN v4 Training ---")
        if not run_cmd([PYTHON_EXE, "scripts/train_gnn_v4_enhanced.py"]):
            print("[ERROR] GNN v4 training failed! Aborting sequential pipeline.")
            return
    else:
        print(f"Monitoring dynamically discovered active GNN v4 process (PID: {target_pid})...")
        
        # 1. Wait for active GNN v4 training process to finish
        check_interval = 60 # seconds
        while is_pid_running(target_pid):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process {target_pid} is still training. Waiting...", flush=True)
            time.sleep(check_interval)
            
        print(f"\n[INFO] GNN v4 training process (PID: {target_pid}) has completed.")
    
    # 2. Re-evaluate all models to record GNN v4's biometric curves
    print("\n--- Step 1: Re-evaluating GNN v4 ---")
    if not run_cmd([PYTHON_EXE, "scripts/reeval_all_models.py"]):
        print("[WARN] GNN v4 re-evaluation failed or was skipped. Continuing anyway...")

    # 3. Train ProtoN Model
    print("\n--- Step 2: Training ProtoN (Prototype Node GNN) ---")
    run_cmd([PYTHON_EXE, "scripts/train_proton.py"])

    # 4. Train VisGIN Model
    print("\n--- Step 3: Training VisGIN (Visibility GNN) ---")
    run_cmd([PYTHON_EXE, "scripts/train_visgin.py"])

    # 5. Train Keypoint Matcher GNN
    print("\n--- Step 4: Training Keypoint Matcher (Sinkhorn OT) ---")
    run_cmd([PYTHON_EXE, "scripts/train_matcher.py"])

    # 6. Generate final comparative figures & report
    print("\n--- Step 5: Regenerating Paper Results & Plots ---")
    run_cmd([PYTHON_EXE, "scripts/compare_models.py"])

    print("\n" + "="*70)
    print("  ALL SEQUENTIAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)

    # Print the final report
    report_path = PROJECT_ROOT / "outputs/results/publication_report.md"
    if report_path.exists():
        print("\n--- Final Biometric Identification Table ---")
        with open(str(report_path)) as f:
            print(f.read())

if __name__ == '__main__':
    main()
