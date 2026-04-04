#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_IVERILOG="$ROOT_DIR/../A3/open_source_flow/local_tools/iverilog_pkg/usr/bin/iverilog"
TOOLS_VVP="$ROOT_DIR/../A3/open_source_flow/local_tools/iverilog_pkg/usr/bin/vvp"

if command -v iverilog >/dev/null 2>&1; then
  IVERILOG_BIN="$(command -v iverilog)"
  VVP_BIN="$(command -v vvp)"
elif [[ -x "$TOOLS_IVERILOG" && -x "$TOOLS_VVP" ]]; then
  IVERILOG_BIN="$TOOLS_IVERILOG"
  VVP_BIN="$TOOLS_VVP"
else
  echo "iverilog/vvp not found" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p results

"$IVERILOG_BIN" -g2012 -o sim/tb_sign_extract.out \
  rtl/sign_extract.sv \
  sim/tb_sign_extract.sv
"$VVP_BIN" sim/tb_sign_extract.out

"$IVERILOG_BIN" -g2012 -o sim/tb_calibration_loop.out \
  rtl/sign_extract.sv \
  rtl/lpf_counter.sv \
  rtl/cal_register.sv \
  model/calibration_behavioral.sv \
  sim/tb_calibration_loop.sv
"$VVP_BIN" sim/tb_calibration_loop.out

rm -f sim/tb_background_calibration.out
"$IVERILOG_BIN" -g2012 -o sim/tb_background_calibration.out \
  rtl/sign_extract.sv \
  rtl/sense_force.sv \
  rtl/lpf_counter.sv \
  rtl/cal_register.sv \
  model/behavioral_sar_raw.sv \
  model/background_calibration_plant.sv \
  sim/tb_background_calibration.sv
"$VVP_BIN" sim/tb_background_calibration.out

echo "Simulation outputs written under Project/results"
