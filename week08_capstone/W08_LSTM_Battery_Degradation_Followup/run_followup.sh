#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"

python "$ROOT/analysis_driver.py" \
  --repo-root "$REPO_ROOT" \
  --output-dir "$ROOT"
