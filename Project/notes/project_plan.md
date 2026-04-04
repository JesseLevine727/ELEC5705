# Project Plan

## Paper

- Ming Ding et al., "A 46 uW 13 b 6.4 MS/s SAR ADC With Background Mismatch and Offset Calibration," IEEE JSSC, 2017

## Goal

Reimplement the key calibration architecture from the paper with a hybrid approach:

- behavioral implementation for most of the ADC and analog correction path
- explicit RTL implementation for the calibration sign-extraction path and its immediate digital control blocks

This matches the project constraint that the sign extraction should not be left as a pure behavioral abstraction.

## What Will Be Behavioral

- 13-bit SAR conversion model with 15 raw comparison cycles
- optional 16th calibration cycle
- redundant code behavior
- two-mode comparator offset model
- DAC mismatch model for the top 5 capacitors
- analog correction effect of DAC trim and comparator trim

## What Will Be Implemented Explicitly

- code-trigger interpretation at the calibration boundary
- D15 / D16 capture interface
- sign extraction logic
- 6-bit bidirectional LPF counter
- calibration register update

## Implementation Partition

### 1. Behavioral ADC / calibration plant

This block emulates:

- comparator-mode offset difference for comparator calibration
- DAC code pair A/B mismatch for DAC calibration
- the effect of trim updates on the underlying error

### 2. Sign extraction RTL

Inputs:

- `cal_mode`: comparator or DAC calibration
- `ab_sel`: A-to-B or B-to-A swap for DAC calibration
- `d15`, `d16`
- `valid`

Outputs:

- `inc_i`
- `dec_i`

Interpretation:

- comparator calibration:
  - `d15,d16 = 0,1` => positive offset difference
  - `d15,d16 = 1,0` => negative offset difference
- DAC calibration:
  - for `A->B`, `0,1` means `V_A > V_B`
  - for `A->B`, `1,0` means `V_A < V_B`
  - for `B->A`, the polarity flips

### 3. LPF counter RTL

- 6-bit bidirectional counter
- increments or decrements according to sign
- only emits output update pulses at saturation
- filters out noisy sign reversals

### 4. Calibration register RTL

- saturating trim register
- one instance for comparator trim
- one instance per calibrated DAC capacitor

## Build Order

1. `sign_extract.sv`
2. `lpf_counter.sv`
3. `cal_register.sv`
4. comparator-offset calibration loop testbench
5. DAC mismatch calibration loop testbench
6. extension to multiple DAC bits and report figures

## Immediate Deliverable

The first runnable project slice should demonstrate:

- a behavioral error source
- real sign extraction
- real LPF filtering
- real trim update
- convergence of comparator offset trim and DAC trim in simulation
