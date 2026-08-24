#!/usr/bin/env bash
# Start Kiro Gateway as a background process.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/kiro-gateway.pid"
LOG_FILE="$PROJECT_DIR/kiro-gateway.log"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Create the virtual environment and install dependencies first:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

is_gateway_process() {
    local pid="$1"
    local command

    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command" == *"$PYTHON_BIN"* && "$command" == *"$PROJECT_DIR/main.py"* ]]
}

if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
        if is_gateway_process "$pid"; then
            echo "Kiro Gateway is already running (PID: $pid)."
            exit 0
        fi
        echo "PID $pid belongs to another process; refusing to overwrite it." >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

cd "$PROJECT_DIR"
nohup "$PYTHON_BIN" "$PROJECT_DIR/main.py" >> "$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Kiro Gateway failed to start. Check the log: $LOG_FILE" >&2
    exit 1
fi

echo "Kiro Gateway started in the background (PID: $pid)."
echo "Log file: $LOG_FILE"
