#!/usr/bin/env bash
# Double-click to start Paper Vault Web UI (macOS)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
if command -v conda &>/dev/null; then
  __conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
  if [ $? -eq 0 ]; then
    eval "$__conda_setup"
    conda activate papervault 2>/dev/null || {
      echo "ERROR: conda env 'papervault' not found. Create it first:"
      echo "  conda create -n papervault python=3.11 -c conda-forge -y"
      echo "  conda activate papervault"
      echo "  pip install -e ."
      read -p "Press Enter to exit..."
      exit 1
    }
  fi
else
  echo "WARNING: conda not found, using system python"
fi

echo "Starting Paper Vault..."
python pv.py serve
