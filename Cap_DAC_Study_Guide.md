# ELEC5705 Study Guide: Capacitive DAC (Cap-DAC) for SAR ADCs

This guide explains how Cap-DACs work, why charge redistribution enables binary search conversion, and what non-idealities control accuracy.

## 1) Role of Cap-DAC in SAR ADC

In a SAR ADC, the Cap-DAC:
1. Samples the input
2. Generates trial analog voltages bit-by-bit
3. Performs subtraction by charge redistribution
4. Works with comparator + SAR logic to converge to final code

Think of it as:
- a memory (holds sampled charge)
- a DAC (creates binary-weighted trial voltages)
- a subtractor (compares residue each bit cycle)

---

## 2) Basic Binary-Weighted Cap Array

Common array:
- `C, C/2, C/4, ...`
- often plus a dummy capacitor to normalize total capacitance

Each capacitor bottom plate can be switched between references (usually `VREF` and `GND`, sometimes common-mode based switching).
Top plate goes to comparator input node.

Key principle:
- Charge on isolated node is conserved during switching.

---

## 3) Charge-Redistribution Operation (Core)

General top-plate node equation after switching:
- `Q_before = Q_after`

Course slide form (subtraction by charge sharing):
- `C(Vin - VREF) = C1(VX - VREF) + C2*VX`
- Solve for node voltage `VX` to get new residue.

Interpretation:
- Switching one binary capacitor changes top-plate voltage by a binary step.
- Comparator checks sign of residue.
- SAR keeps/clears bit and proceeds to next smaller step.

---

## 4) SAR Conversion Sequence with Cap-DAC

Typical steps:

1. **Sample phase**
- Array samples `Vin` (and often differential complement in fully differential design).

2. **MSB trial**
- Switch MSB capacitor to reference state producing `+/- VREF/2` equivalent step.
- Comparator decides whether residue is above or below zero.
- SAR stores MSB.

3. **Next bit trial**
- Apply next step (`VREF/4` equivalent), preserving prior bit decision.
- Compare again.

4. **Repeat to LSB**
- After `N` comparisons, all bits resolved.

This is binary search in analog domain.

---

## 5) Important Equations

## 5.1 LSB size
- `LSB = VFS / 2^N`

For an ideal SAR with proper scaling:
- Bit `k` contributes binary-weighted step in residue (`VREF/2`, `VREF/4`, ...).

## 5.2 Charge conservation

Generic node update:
- `sum(Ci*(Vnode - Vi_after)) = constant` (from sampled initial charge)

Use this relation to derive each trial node voltage.

## 5.3 Thermal noise from sampling network

At sampled node, `kT/C` still applies using effective sampling capacitance:
- `v_n,rms^2 approx kT/C_total`

Larger total capacitance lowers sampling noise but increases switching energy and settling demand.

## 5.4 Switching energy trend

Dynamic energy per conversion scales with:
- capacitance size
- number of switched transitions
- `VREF^2`

Hence many SARs use monotonic/split switching to reduce energy.

---

## 6) Non-Idealities That Matter Most

1. Capacitor mismatch
- Breaks ideal binary weights
- Causes DNL/INL error and harmonic distortion

2. Reference settling / driver impedance
- If DAC node or reference does not settle each bit cycle, comparison is wrong

3. Parasitic capacitance at top plate
- Alters effective bit weights and residue gain

4. Switch non-idealities
- Charge injection, feedthrough, Ron nonlinearity

5. Comparator kickback into DAC node
- Disturbs residue before or during decision

6. Incomplete common-mode control (differential arrays)
- Converts common-mode disturbances into differential error

---

## 7) Layout/Architecture Choices

Binary-weighted single array:
- Simple concept
- Large total capacitance for high N

Split capacitor array:
- Reduces area and load
- Needs bridge-cap calibration/accuracy management

Differential Cap-DAC:
- Better even-order distortion and common-mode rejection
- More switches/capacitance/complexity

Calibration (digital/background):
- Corrects mismatch-induced linearity errors
- Adds logic/time complexity

---

## 8) Fast Exam Workflow for Cap-DAC Questions

1. Identify sampling phase and node that is charge-conserved
2. Write `Q_before = Q_after`
3. Solve residue node voltage after each bit trial
4. Map comparator result to SAR bit decision
5. Repeat binary-step logic to final code
6. Comment on dominant error source (mismatch, settling, parasitic, kickback)

---

## 9) High-Probability Question Types

1. Derive residue node voltage after one or two bit trials using charge conservation.
2. Explain how SAR binary search is implemented physically by capacitor switching.
3. Show how capacitor mismatch maps to INL/DNL and spurs.
4. Compare binary-weighted vs split Cap-DAC tradeoffs (area, speed, energy, accuracy).

