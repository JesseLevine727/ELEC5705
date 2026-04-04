# Project RTL Synthesis Experiment

This directory synthesizes only the explicitly implemented digital calibration path from the course project:

- `sense_force`
- `sign_extract`
- six copies of the 6-bit bidirectional `lpf_counter`
- six copies of the 7-bit saturating `cal_register`

The behavioral ADC and analog calibration plant are intentionally excluded.

The synthesis scripts are now project-local. They live under `Project/synthesis/`, and by default they look for local Yosys tooling in `Project/synthesis/local_tools/`. If that directory is absent, they fall back to the existing `A3/synthesis_experiment/local_tools/` tool package.

## Outputs

Running `bash run_local_flow.sh` generates:

- `project_calibration_digital_top_netlist.v`
- `project_calibration_digital_top.json`
- `project_calibration_digital_top.dot`
- `synthesis.log`
- `project_calibration_digital_top_hierarchy.svg`

## Notes

The wrapper top-level `project_calibration_digital_top.sv` fans the decoded sign request to one of six calibration channels:

- channels `0..4`: DAC-capacitor trim loops
- channel `5`: comparator trim loop

The SVG file is a clean block-level hierarchy view of the synthesized digital path. The gate-level structure remains available in the generated Verilog netlist and DOT file.

## Per-block synthesis

Run:

```bash
bash run_block_synthesis.sh
```

This generates separate synthesized outputs for the implemented RTL blocks:

- `sense_force`
- `sign_extract`
- `lpf_counter`
- `cal_register`

Outputs are written under `block_outputs/` as:

- per-block synthesized netlists
- per-block JSON netlists
- per-block DOT graphs
- per-block SVG schematic-style views
- per-block synthesis logs
