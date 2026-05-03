#!/bin/bash
# Watchdog: monitors phase2 script, auto-fixes and restarts on failure.

export PATH="/home/vanh/miniconda3/envs/t2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
cd "$(dirname "$0")"

RESULTS_FILE="phase2_results.json"
LOGDIR="run_logs"
WATCHDOG_LOG="watchdog.log"
RUNNER="run_smart.sh"
MAX_RESTARTS=20

log() { echo "[watchdog $(date '+%H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

fix_json() {
    python3 - <<'PYEOF'
import json, os, sys
path = os.environ.get("RESULTS_FILE", "phase2_results.json")
try:
    raw = open(path).read()
    # Try parsing as-is first
    data = json.loads(raw)
    # Check for duplicate entries and remove them
    seen = {}
    clean = []
    for e in data.get("results", []):
        key = (e.get("task"), e.get("strategy"))
        if key not in seen:
            seen[key] = True
            clean.append(e)
    data["results"] = clean
    json.dump(data, open(path, "w"), indent=2)
    print(f"JSON OK: {len(clean)} entries")
except json.JSONDecodeError:
    # Find last valid closing brace
    idx = raw.rfind("}")
    if idx == -1:
        data = {"results": []}
    else:
        try:
            data = json.loads(raw[:idx+1])
            # Deduplicate
            seen = {}
            clean = []
            for e in data.get("results", []):
                key = (e.get("task"), e.get("strategy"))
                if key not in seen:
                    seen[key] = True
                    clean.append(e)
            data["results"] = clean
        except Exception:
            data = {"results": []}
    json.dump(data, open(path, "w"), indent=2)
    print(f"JSON FIXED: {len(data['results'])} entries")
PYEOF
}

count_done() {
    python3 -c "
import json
try:
    data = json.load(open('$RESULTS_FILE'))
    print(len(data.get('results', [])))
except Exception:
    print(0)
"
}

restart_count=0
while true; do
    log "Starting $RUNNER (restart #$restart_count)"
    RESULTS_FILE="$RESULTS_FILE" fix_json

    bash "$RUNNER" >> phase2_full.log 2>&1
    EXIT=$?

    done=$(count_done)
    log "Process exited (code=$EXIT). Entries saved: $done/12"

    if [ "$done" -ge 12 ]; then
        log "All 12 experiments complete! Results in $RESULTS_FILE"
        break
    fi

    if [ "$EXIT" -eq 0 ]; then
        log "Script exited normally but only $done/12 done — something wrong, stopping."
        break
    fi

    restart_count=$((restart_count + 1))
    if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
        log "Too many restarts ($MAX_RESTARTS). Stopping."
        break
    fi

    log "Fixing JSON and restarting in 5s..."
    RESULTS_FILE="$RESULTS_FILE" fix_json
    sleep 5
done
