#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# FLEET SAFE VLA — Fleet Trinity Dashboard Launcher
# Starts: Python Backend (8000) + Next.js Dashboard (3000)
# ═══════════════════════════════════════════════════════════════════
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/scripts/.logs"
mkdir -p "$LOG_DIR"

echo "  ┌─ FLEET TRINITY DASHBOARD ─────────────────────────┐"

# Backend
if ! lsof -ti:8000 > /dev/null 2>&1; then
  echo "  │ Starting Backend on :8000...                     │"
  cd "$PROJECT_DIR"
  python3 -m uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload \
    > "$LOG_DIR/backend.log" 2>&1 &
fi

# Next.js
if ! lsof -ti:3000 > /dev/null 2>&1; then
  echo "  │ Starting Next.js on :3000...                     │"
  cd "$PROJECT_DIR/fleet_trinity_dashboard"
  npm run dev > "$LOG_DIR/trinity.log" 2>&1 &
fi

echo "  └───────────────────────────────────────────────────┘"
sleep 4
open -a "Google Chrome" "http://localhost:3000"
echo "  ✓ Fleet Trinity Dashboard → http://localhost:3000"
