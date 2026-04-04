#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$ROOT_DIR/local_tools"
FALLBACK_TOOLS_DIR="$(cd "$ROOT_DIR/../../A3/synthesis_experiment/local_tools" 2>/dev/null && pwd || true)"
if [[ ! -d "$TOOLS_DIR/yosys_pkg" && -n "$FALLBACK_TOOLS_DIR" ]]; then
  TOOLS_DIR="$FALLBACK_TOOLS_DIR"
fi
YOSYS_PKG_DIR="$TOOLS_DIR/yosys_pkg"

YOSYS_BIN="$YOSYS_PKG_DIR/usr/bin/yosys"

if [[ ! -x "$YOSYS_BIN" ]]; then
  echo "Missing local yosys binary at $YOSYS_BIN" >&2
  exit 1
fi

export ABCEXTERNAL="${YOSYS_PKG_DIR}/usr/bin/yosys-abc"

cd "$ROOT_DIR"
"$YOSYS_BIN" -s synth.ys | tee synthesis.log
