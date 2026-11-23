#!/usr/bin/env bash
# Mouna quality checks: unit tests, per-sign quality gates, regression
# snapshot, and a frontend build. Run from anywhere:
#   scripts/run_checks.sh            # everything
#   scripts/run_checks.sh --fast     # tests only (skip the frontend build)
# After an INTENDED change to sign data, refresh the snapshot with:
#   MOUNA_UPDATE_GOLDEN=1 ./venv/bin/python -m pytest tests/test_sign_quality.py -k golden
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== Python tests (logic, sign quality gates, regression snapshot)"
./venv/bin/python -m pytest -o addopts="" tests -q
if [[ "${1:-}" != "--fast" ]]; then
  echo "== Frontend build"
  (cd app/frontend && npx ng build --configuration development >/tmp/mouna_ng_build.log 2>&1) \
    && echo "frontend build OK" || { echo "frontend build FAILED (see /tmp/mouna_ng_build.log)"; exit 1; }
fi
echo "All checks passed."
