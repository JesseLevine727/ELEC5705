# Background Calibration Status

Current project model:

- real RTL:
  - `sign_extract.sv`
  - `sense_force.sv`
  - `lpf_counter.sv`
  - `cal_register.sv`
- behavioral model:
  - behavioral 13-bit conversion source with a simplified 15-bit redundant raw-code encoding
  - sparse calibration activation using paper-like X/Y code gating
  - comparator offset plant
  - five DAC-mismatch plants
  - occasional wrong-sign observations to emulate noisy calibration

Key result from `tb_background_calibration.sv`:

- total convergence cycle: `318708`
- active calibration events: `4361`
- active-event rate: about `1.37%`

Per-loop convergence:

- comparator offset loop: `171056` cycles
- DAC channel 0: `318708` cycles
- DAC channel 1: `300672` cycles
- DAC channel 2: `287792` cycles
- DAC channel 3: `237604` cycles
- DAC channel 4: `205441` cycles

Interpretation:

- this is now in the same order of magnitude as the paper's reported background convergence time of roughly `400k` cycles
- the large change from the earlier trivial model comes from:
  - sparse trigger-code activation
  - separate calibration channels
  - reset-on-update LPF behavior
  - occasional wrong-sign observations
  - raw calibration triggers now coming from a behavioral ADC conversion result instead of a synthetic random raw-code source

Digital LPF note:

- the paper's LPF is implemented as a 6-bit bidirectional counter
- the current RTL follows that structure directly in `rtl/lpf_counter.sv`

Current limitations:

- the trigger table is paper-like and centered around midscale, but still simplified
- the analog correction plant is behavioral and linear

Next refinement:

- replace the simplified redundant raw-code packing with a more exact 15-cycle redundancy mapping from the paper's conversion schedule
