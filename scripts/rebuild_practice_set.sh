#!/usr/bin/env bash
# Turn new review approvals into practice signs, end to end:
#   1. rebuild references from approved clips (majority-vote CP targets)
#   2. put back signs/face data a rebuild can't produce
#   3. re-pick each sign's example video against its reference
#   4. hand paths + hand counts, then 3D display clips
#   5. quality checks (gates + regression snapshot)
# A copy of the reference file is kept in data/references/backups/.
# If step 5 reports regressions you EXPECT (e.g. a sign's grading improved),
# refresh the snapshot:  MOUNA_UPDATE_GOLDEN=1 ./venv/bin/python -m pytest tests/test_sign_quality.py -k golden
set -euo pipefail
cd "$(dirname "$0")/.."
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p data/references/backups
cp data/references/sign_references.pkl "data/references/backups/sign_references.$stamp.pkl"
cp data/references/example_videos.json "data/references/backups/example_videos.$stamp.json"
echo "== 1/5 references";   ./venv/bin/python scripts/build_sign_references_v2.py | grep -E "References built|Saved"
echo "== 2/5 merge";        ./venv/bin/python scripts/merge_reference_extras.py "data/references/backups/sign_references.$stamp.pkl"
echo "== 3/5 examples";     ./venv/bin/python scripts/pick_example_videos.py 12 2>/dev/null | grep -E "Practice-ready|Saved"
echo "== 4/5 paths + 3D";   ./venv/bin/python scripts/pick_example_videos.py --paths-only 2>/dev/null | grep -E "Saved"
                            ./venv/bin/python scripts/choose_display_clips.py 2>/dev/null | grep -E "Saved"
echo "== 5/5 checks";       scripts/run_checks.sh --fast
