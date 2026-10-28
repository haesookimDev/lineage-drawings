#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export ALLOW_INSECURE_LOCAL=1
export DATA_DIR="${DATA_DIR:-$PWD/data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/lineage-cache}"
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
