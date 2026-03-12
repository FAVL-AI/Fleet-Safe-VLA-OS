#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# FLEET SAFE VLA — Master Launcher
# Starts: Python Backend (8000) + Fleet Trinity (3000) + Command Center (8080)
# ═══════════════════════════════════════════════════════════════════
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/scripts/.logs"
mkdir -p "$LOG_DIR"

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║   F.L.E.E.T. SAFE VLA — LAUNCH SEQUENCE INITIATED   ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Python Backend (FastAPI + Uvicorn) ─────────────────────
echo "  [1/3] Starting Python Backend on :8000..."
if lsof -ti:8000 > /dev/null 2>&1; then
  echo "        → Already running on :8000, skipping."
else
  cd "$PROJECT_DIR"
  python3 -m uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload \
    > "$LOG_DIR/backend.log" 2>&1 &
  echo "        → PID: $!"
fi

# ── 2. Fleet Trinity Dashboard (Next.js) ──────────────────────
echo "  [2/3] Starting Fleet Trinity Dashboard on :3000..."
if lsof -ti:3000 > /dev/null 2>&1; then
  echo "        → Already running on :3000, skipping."
else
  cd "$PROJECT_DIR/fleet_trinity_dashboard"
  npm run dev > "$LOG_DIR/trinity.log" 2>&1 &
  echo "        → PID: $!"
fi

# ── 3. Command Center (Static File Server) ────────────────────
echo "  [3/3] Starting Command Center on :8080..."
if lsof -ti:8080 > /dev/null 2>&1; then
  echo "        → Already running on :8080, skipping."
else
  cd "$PROJECT_DIR"
  python3 serve_cors.py > "$LOG_DIR/command_center.log" 2>&1 &
  echo "        → PID: $!"
fi

# ── Wait for services to boot ─────────────────────────────────
echo ""
echo "  Waiting for services to initialize..."
sleep 4

# ── Open in Chrome ────────────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Fleet Trinity Dashboard:  http://localhost:3000    │"
echo "  │  Command Center:           http://localhost:8080    │"
echo "  │  Backend API:              http://localhost:8000    │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
echo "  Opening dashboards in Chrome..."

open -a "Google Chrome" "http://localhost:3000"
sleep 1
open -a "Google Chrome" "http://localhost:8080/fastbot_command_center.html"

echo ""
echo "  ✓ All systems operational. Use stop_fleet.sh to shutdown."
echo ""
