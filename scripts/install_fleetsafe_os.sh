#!/bin/bash
set -e

echo "Installing FleetSafe Core (formerly DimOS)..."
cd "$(dirname "$0")/../fleetsafe_core"
uv pip install -e ".[all]" || python3 -m pip install -e ".[all]"

echo "Installing FleetSafe VLA Extensions..."
cd ../fleetsafe_vla
uv pip install -e . || python3 -m pip install -e .

echo "Installation Complete."
