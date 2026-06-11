#!/usr/bin/env bash
# Start Paper Vault Web UI (Linux)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if command -v conda &>/dev/null; then
  __conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
  if [ $? -eq 0 ]; then
    eval "$__conda_setup"
    conda activate papervault 2>/dev/null || {
      echo "ERROR: conda env 'papervault' not found. Create it first."
      exit 1
    }
  fi
fi

echo "Starting Paper Vault..."
python pv.py serve
