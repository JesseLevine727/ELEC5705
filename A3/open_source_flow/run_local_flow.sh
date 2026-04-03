#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IVERILOG="$SCRIPT_DIR/local_tools/iverilog_pkg/usr/bin/iverilog"
VVP="$SCRIPT_DIR/local_tools/iverilog_pkg/usr/bin/vvp"

"$IVERILOG" -g2012 -o "$SCRIPT_DIR/async_sar_tb" \
  "$SCRIPT_DIR/tb_async_sar.sv" \
  "$SCRIPT_DIR/sar_logic_async.sv" \
  "$SCRIPT_DIR/strongarm_comp_model.sv"

"$VVP" "$SCRIPT_DIR/async_sar_tb"
python3 "$SCRIPT_DIR/analyze_enob.py"
