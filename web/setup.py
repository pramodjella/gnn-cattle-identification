#!/usr/bin/env python3
"""
Quick setup script for the Cattle ID Web Platform.
Copies .env.example → .env and installs backend requirements.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"

print("=" * 60)
print("  Cattle Biometric ID Platform – Setup")
print("=" * 60)

# 1. Create .env from example
env_src = BACKEND / ".env.example"
env_dst = BACKEND / ".env"
if not env_dst.exists():
    shutil.copy(env_src, env_dst)
    print("✅ Created backend/.env from .env.example")
    print("   Edit backend/.env to set your DATABASE_URL if needed.")
else:
    print("ℹ️  backend/.env already exists – skipping.")

# 2. Install backend requirements
print("\n📦 Installing backend Python requirements…")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(BACKEND / "requirements.txt")],
    check=False
)
if result.returncode == 0:
    print("✅ Backend requirements installed.")
else:
    print("⚠️  Some backend packages failed. Check errors above.")

print("""
─────────────────────────────────────────────
Next steps:

1. Start PostgreSQL with pgvector (Docker recommended):
   cd web
   docker-compose up -d postgres

2. Start the FastAPI backend:
   cd web/backend
   uvicorn main:app --reload --port 8000

3. Start the React frontend:
   cd web/frontend
   npm run dev

4. Open http://localhost:5173

─────────────────────────────────────────────
Optional: Train the GNN model first:
   python scripts/05_train.py
   (The web app works WITHOUT training using SIFT fallback)
─────────────────────────────────────────────
""")
