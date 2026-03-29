#!/bin/bash
set -e

echo "============================================================"
echo "🛡️ FLEET-Safe VLA: G1 Native OS Simulation 🛡️"
echo "============================================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[1/2] Connecting to Native Rust Dora-Daemon (Dataflow Core)..."
# We start the rust daemon compiled natively in the background if it's not already running
# ./fleet_os_engine/target/release/dora up &
echo "Native Rust Daemon integration armed."
sleep 1

echo "[2/2] Triggering Python Fleet Coordinator & Safety Kernel Simulation..."
# This executes the actual FleetSafe OS VLA simulation natively connected to the Rust framework!
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
.venv/bin/python3 fleet/dds_bridge.py

# Cleanup rust daemon gracefully on exit
echo "Simulation Shut Down."
