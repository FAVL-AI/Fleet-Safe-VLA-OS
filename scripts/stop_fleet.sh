#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# FLEET SAFE VLA — Graceful Shutdown
# Stops all services on ports 3000, 8000, 8080
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║        F.L.E.E.T. — SHUTDOWN SEQUENCE                ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""

for PORT in 3000 8000 8080; do
  PIDS=$(lsof -ti:$PORT 2>/dev/null)
  if [ -n "$PIDS" ]; then
    echo "  Stopping port $PORT (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill -TERM 2>/dev/null
    sleep 1
    # Force kill any remaining
    REMAINING=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$REMAINING" ]; then
      echo "  Force-killing port $PORT..."
      echo "$REMAINING" | xargs kill -9 2>/dev/null
    fi
    echo "  ✓ Port $PORT stopped."
  else
    echo "  · Port $PORT — not running."
  fi
done

echo ""
echo "  ✓ All services stopped."
echo ""
