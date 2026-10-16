#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -U pip
  pip install -e ".[dev]"
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> AgentThree Enterprise RAG demo (local, zero API / cloud cost)"
agentthree-rag demo
echo
echo "==> Offline RAG evaluation"
agentthree-rag eval
echo
echo "==> pytest"
pytest -q
