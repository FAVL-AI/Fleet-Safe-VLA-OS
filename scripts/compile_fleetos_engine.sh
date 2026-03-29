#!/bin/bash
set -e

echo "[1/4] Ensuring Rust toolchain is installed locally..."
if ! command -v cargo &> /dev/null; then
  echo "Installing rustup silently..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source $HOME/.cargo/env
else
  echo "Cargo is already installed."
fi

echo "[2/4] Cloning Dora-RS core engine (FleetOS backend)..."
rm -rf fleet_os_engine || true
git clone --depth 1 https://github.com/dora-rs/dora.git fleet_os_engine

echo "[3/4] Preparing integration..."
cd fleet_os_engine
rm -rf .git
find . -type f -name "*.md" -exec rm -f {} +

echo "[4/4] Starting heavy Rust Compilation (This isolates dora runtime and daemon)..."
source $HOME/.cargo/env
# Dora-rs workspaces require specific targeted builds
cargo build --release -p dora-runtime
cargo build --release -p dora-daemon
echo "Compilation complete."
