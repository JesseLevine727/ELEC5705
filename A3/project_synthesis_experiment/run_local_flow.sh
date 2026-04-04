#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="/home/elfo/Documents/ELEC5705/A3/synthesis_experiment/local_tools"
YOSYS_PKG_DIR="$TOOLS_DIR/yosys_pkg"

YOSYS_BIN="$YOSYS_PKG_DIR/usr/bin/yosys"

if [[ ! -x "$YOSYS_BIN" ]]; then
  echo "Missing local yosys binary at $YOSYS_BIN" >&2
  exit 1
fi

export ABCEXTERNAL="${YOSYS_PKG_DIR}/usr/bin/yosys-abc"

cd "$ROOT_DIR"
"$YOSYS_BIN" -s synth.ys | tee synthesis.log
