#!/usr/bin/env bash
# End-to-end HumanEval: generate -> execute (sandboxed) -> report.
#
#   ./scripts/run-humaneval.sh            # thinking off
#   ./scripts/run-humaneval.sh think      # thinking on (slow: ~20 min)
#
# Execution happens inside a `--network none` container because this runs
# model-generated code. The repo is mounted read-only.
set -euo pipefail

MODE="${1:-nothink}"
[ "$MODE" = "think" ] || MODE="nothink"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GB10_PYTHON:-python3}"

mkdir -p "$ROOT/results" "$ROOT/data"

if ! find "$ROOT/data/humaneval" -name '*.parquet' 2>/dev/null | grep -q .; then
  echo "== fetching HumanEval =="
  hf download openai/openai_humaneval --repo-type dataset \
    --local-dir "$ROOT/data/humaneval"
fi

echo "== generating ($MODE) =="
( cd "$ROOT/bench/humaneval" && $PY generate.py ${1:-} )

echo
echo "== executing in sandbox =="
docker run --rm --network none -v "$ROOT:/w:ro" -w /tmp python:3.12-slim \
  python /w/bench/humaneval/execute.py "/w/results/gen_${MODE}.json" \
  2>/dev/null > "$ROOT/results/exec_${MODE}.json"

echo
echo "== report =="
$PY "$ROOT/bench/humaneval/report.py" \
  "$ROOT/results/exec_${MODE}.json" "$ROOT/results/gen_${MODE}.json"
