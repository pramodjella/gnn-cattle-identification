"""
Wrapper script for GNN and Hybrid Attention Visualization
==========================================================
Calls scripts/visualize_attention.py to generate explainability plots.
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    script_path = PROJECT_ROOT / 'scripts' / 'visualize_attention.py'
    args = sys.argv[1:]
    
    # Default to hybrid if not specified
    if not any(arg.startswith('--model') for arg in args):
        args.extend(['--model', 'hybrid'])
        
    cmd = [sys.executable, str(script_path)] + args
    print(f"[INFO] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
