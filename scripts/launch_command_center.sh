#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# FLEET SAFE VLA — Command Center Launcher
# Starts: Static File Server (8080) for fastbot_command_center.html
# ═══════════════════════════════════════════════════════════════════
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/scripts/.logs"
mkdir -p "$LOG_DIR"

echo "  ┌─ COMMAND CENTER ──────────────────────────────────┐"

# Backend (needed for WebSocket + API)
if ! lsof -ti:8000 > /dev/null 2>&1; then
  echo "  │ Starting Backend on :8000...                     │"
  cd "$PROJECT_DIR"
  python3 -m uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload \
    > "$LOG_DIR/backend.log" 2>&1 &
fi

# Static server
if ! lsof -ti:8080 > /dev/null 2>&1; then
  echo "  │ Starting Static Server on :8080...               │"
  cd "$PROJECT_DIR"
  python3 serve_cors.py > "$LOG_DIR/command_center.log" 2>&1 &
fi

echo "  └───────────────────────────────────────────────────┘"
sleep 3
open -a "Google Chrome" "http://localhost:8080/fastbot_command_center.html"
echo "  ✓ Command Center → http://localhost:8080/fastbot_command_center.html"
