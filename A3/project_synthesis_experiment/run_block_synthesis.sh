#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/block_outputs"
YOSYS_PKG_DIR="/home/elfo/Documents/ELEC5705/A3/synthesis_experiment/local_tools/yosys_pkg"
YOSYS_BIN="$YOSYS_PKG_DIR/usr/bin/yosys"

mkdir -p "$OUT_DIR"

if [[ ! -x "$YOSYS_BIN" ]]; then
  echo "Missing local yosys binary at $YOSYS_BIN" >&2
  exit 1
fi

export ABCEXTERNAL="${YOSYS_PKG_DIR}/usr/bin/yosys-abc"

run_one() {
  local module="$1"
  local title="$2"
  local script="$OUT_DIR/${module}.ys"

  cat > "$script" <<EOF
read_verilog -sv /home/elfo/Documents/ELEC5705/Project/rtl/${module}.sv
hierarchy -top ${module}
proc
opt
fsm
opt
memory
opt
techmap
opt
abc -g simple
opt_clean
stat
write_json $OUT_DIR/${module}.json
write_verilog -noattr $OUT_DIR/${module}_netlist.v
show -format dot -prefix $OUT_DIR/${module}
EOF

  "$YOSYS_BIN" -s "$script" | tee "$OUT_DIR/${module}.log"
  python3 "$ROOT_DIR/render_dot_to_svg.py" "$OUT_DIR/${module}.dot" "$OUT_DIR/${module}.svg" "$title"
}

run_one "sense_force" "sense_force Synthesized View"
run_one "sign_extract" "sign_extract Synthesized View"
run_one "lpf_counter" "lpf_counter Synthesized View"
run_one "cal_register" "cal_register Synthesized View"
