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
  - two-mode comparator offset plant
  - five DAC-mismatch plants built from calibrated bit weights
  - `D15` and `D16` generated from actual behavioral DAC/comparator comparisons
  - comparator noise / comparison perturbation around the calibration boundary

Key result from `tb_background_calibration.sv`:

- total convergence cycle: `384243`
- active calibration events: `5257`
- active-event rate: about `1.37%`

Per-loop convergence:

- comparator offset loop: `299008` cycles
- DAC channel 0: `384243` cycles
- DAC channel 1: `346060` cycles
- DAC channel 2: `252461` cycles
- DAC channel 3: `219268` cycles
- DAC channel 4: `190316` cycles

Interpretation:

- this is now very close to the paper's reported background convergence time of roughly `400k` cycles
- the large change from the earlier trivial model comes from:
  - sparse trigger-code activation
  - separate calibration channels
  - reset-on-update LPF behavior
  - raw calibration triggers now coming from a behavioral ADC conversion result instead of a synthetic random raw-code source
  - `D15/D16` sign formation now coming from comparator-mode and DAC-code behavioral comparisons instead of a placeholder signed error variable

Digital LPF note:

- the paper's LPF is implemented as a 6-bit bidirectional counter
- the current RTL follows that structure directly in `rtl/lpf_counter.sv`

Sign-extraction note:

- the implemented `sign_extract.sv` maps `D15/D16` into the correction direction required by the trim hardware
- this is the key non-behavioral block requested for the project
- the sign convention is therefore tied to the correction orientation, not to a standalone mathematical sign label

Current limitations:

- the trigger table is paper-like and centered around midscale, but still simplified
- the analog correction plant is behavioral and linear

Next refinement:

- replace the simplified redundant raw-code packing with a more exact 15-cycle redundancy mapping from the paper's conversion schedule
