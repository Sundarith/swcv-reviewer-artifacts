#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Connection settings belong in a reviewed JSON file.
if [[ $# -ne 3 ]]; then
  echo "Usage: bash examples/run_naive_rag20.sh CONFIG.json PREPARED_INPUTS.jsonl NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python -m swcv.inference \
  --config "$1" --inputs "$2" --out "$3"
