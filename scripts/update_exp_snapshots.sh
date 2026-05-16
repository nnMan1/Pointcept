#!/bin/bash
# Updejtuje code/ snapshot u svakom insseg eksperimentu na aktuelne metric/engines fajlove.
# Pokrene se iz /home (repo root).
set -e

ROOT=/home
SRC_METRICS="$ROOT/pointcept/utils/metrics.py"
SRC_MATCHER_DIR="$ROOT/pointcept/utils/matcher"
SRC_UTILS_INIT="$ROOT/pointcept/utils/__init__.py"
SRC_TEST_PY="$ROOT/pointcept/engines/test.py"
SRC_EVALUATOR="$ROOT/pointcept/engines/hooks/evaluator.py"

UPDATED=0
SKIPPED=0
FAILED=0
LOG="$ROOT/scripts/update_exp_snapshots.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

for exp in $(find "$ROOT/exp" -maxdepth 2 -mindepth 2 -type d | sort); do
  name=${exp#$ROOT/exp/}
  type=$(echo "$name" | sed 's|.*/||' | grep -oE "^(insseg|semseg|cls)" || echo "?")
  [ "$type" != "insseg" ] && { log "SKIP (not insseg): $name"; SKIPPED=$((SKIPPED+1)); continue; }
  [ ! -f "$exp/model/model_best.pth" ] && { log "SKIP (no model): $name"; SKIPPED=$((SKIPPED+1)); continue; }
  [ ! -f "$exp/code/pointcept/engines/test.py" ] && { log "SKIP (no test.py snapshot): $name"; SKIPPED=$((SKIPPED+1)); continue; }

  code="$exp/code/pointcept"
  utils="$code/utils"
  hooks="$code/engines/hooks"

  log "UPDATE: $name"
  {
    mkdir -p "$utils/matcher" "$hooks"
    cp -f "$SRC_METRICS" "$utils/metrics.py"
    cp -rf "$SRC_MATCHER_DIR/." "$utils/matcher/"
    cp -f "$SRC_UTILS_INIT" "$utils/__init__.py"
    cp -f "$SRC_TEST_PY" "$code/engines/test.py"
    cp -f "$SRC_EVALUATOR" "$hooks/evaluator.py"
    # obrisemo stari metric.py ako postoji
    [ -f "$utils/metric.py" ] && rm -f "$utils/metric.py"
    UPDATED=$((UPDATED+1))
  } || { log "FAIL: $name"; FAILED=$((FAILED+1)); }
done

log ""
log "=== SUMMARY ==="
log "Updated: $UPDATED"
log "Skipped: $SKIPPED"
log "Failed : $FAILED"
