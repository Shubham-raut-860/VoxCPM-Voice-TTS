#!/usr/bin/env bash
set -euo pipefail

echo "==> Voice Agent API — Setup"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example — edit before running"
fi

mkdir -p storage/voices storage/outputs storage/loras

echo ""
echo "==> Setup complete"
echo "==> Run: uvicorn app.main:app --reload --port 8000"
echo "==> Docs: http://localhost:8000/docs"
