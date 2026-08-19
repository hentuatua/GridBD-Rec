#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-Video_Games}"

case "$DATASET" in
  Video_Games|Musical_Instruments|Baby_Products)
    ;;
  *)
    echo "[error] Unsupported dataset: $DATASET"
    echo "Usage: bash run.sh {Video_Games|Musical_Instruments|Baby_Products} [extra args...]"
    exit 1
    ;;
esac

shift || true

python "$ROOT/main.py" --dataset "$DATASET" "$@"
