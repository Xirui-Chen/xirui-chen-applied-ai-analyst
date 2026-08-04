#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv_w07"
REQUIREMENTS="$ROOT/requirements.txt"
STAMP="$VENV/.requirements.sha256"
PORT="${PORT:-8501}"

choose_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    echo "Python 3.10+ is required but was not found." >&2
    exit 1
  fi
}

PYTHON_BIN="${PYTHON_BIN:-$(choose_python)}"

if [[ ! -d "$VENV" ]]; then
  echo "[1/4] Creating virtual environment with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

REQ_HASH="$(REQUIREMENTS_PATH="$REQUIREMENTS" python - <<'PY'
from pathlib import Path
import hashlib
import os
path = Path(os.environ["REQUIREMENTS_PATH"])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)"

if [[ ! -f "$STAMP" ]] || [[ "$(cat "$STAMP")" != "$REQ_HASH" ]]; then
  echo "[2/4] Installing dashboard dependencies"
  python -m pip install --upgrade pip
  python -m pip install -r "$REQUIREMENTS"
  printf '%s' "$REQ_HASH" > "$STAMP"
else
  echo "[2/4] Dependencies already installed"
fi

echo "[3/4] Validating sample-data contracts"
python "$ROOT/scripts/validate_data.py"

echo "[4/4] Launching Streamlit at http://localhost:$PORT"
cd "$ROOT"
exec streamlit run app.py \
  --server.address=localhost \
  --server.port="$PORT" \
  --server.headless=true
