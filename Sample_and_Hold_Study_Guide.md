# ELEC5705 Study Guide: Sample and Hold (S/H)

This guide breaks down how an ADC sample-and-hold works, what limits it, and what exam questions usually target.

## 1) What the Sample-and-Hold Does

The S/H converts a continuously changing input `vin(t)` into a nearly constant value during conversion.

Without S/H:
- During multi-step conversion (especially SAR/pipeline), input keeps moving.
- Comparator decisions are inconsistent from bit to bit.

With S/H:
- Sample phase tracks input.
- Hold phase freezes one value for accurate bit decisions.

Core idea:
- Time-domain "freezing" of amplitude.

---

## 2) Basic Circuit and Phases

Minimal switched-capacitor S/H:
- Switch `S` (MOS switch)
- Hold capacitor `CH`
- Buffer (optional, to isolate loading)

Two phases:
1. Track/Sample (`S` ON):
   - `vH approx vin`
   - Capacitor follows input.
2. Hold (`S` OFF):
   - Charge on `CH` should remain constant.
   - ADC backend reads fixed held voltage.

Timing terms:
- `tacq`: acquisition time in track mode
- `thold`: required hold time during conversion
- `taperture`: effective sampling instant uncertainty window

---

## 3) First-Order Equations You Need

## 3.1 Acquisition settling

For source resistance `RS` + switch on-resistance `RON` charging `CH`:
- `tau = (RS + RON) * CH`
- Residual settling error after `tacq`:
- `e_settle = exp(-tacq/tau)` (fraction of step)

N-bit target rule:
- To settle within `0.5 LSB` for full-scale step:
- `e_settle <= 1/2^(N+1)`
- Equivalent:
- `tacq >= tau * ln(2^(N+1)) approx 0.693*(N+1)*tau`

## 3.2 Thermal sampling noise (kT/C)

After sampling:
- `v_n,rms^2 = kT/CH`
- `v_n,rms = sqrt(kT/CH)`

Design implication:
- Larger `CH` lowers noise but increases settling time and power.

## 3.3 Droop in hold mode

If leakage current `Ileak` discharges hold capacitor:
- `dV/dt = Ileak / CH`
- Over hold interval `thold`:
- `DeltaV_droop approx Ileak * thold / CH`

Must keep droop << `0.5 LSB`.

## 3.4 Aperture jitter limit (for sinusoidal input)

Jitter-induced SNR:
- `SNR_jitter = -20log10(2*pi*fin*sigma_t)`

Where:
- `fin` is input tone frequency
- `sigma_t` is RMS sampling-time jitter

At high `fin`, jitter dominates quickly.

---

## 4) Real Non-Idealities (High-Yield)

From course notes, key S/H concerns:
- Bandwidth
- Settling error
- Feedthrough
- Charge injection
- DC droop

Breakdown:

1. Bandwidth limit
- Finite track bandwidth attenuates high-frequency input.
- Causes gain error and distortion.

2. Incomplete settling
- Not enough `tacq` -> held value lags true input.
- Appears as code-dependent dynamic error.

3. Charge injection
- When MOS switch turns off, channel charge dumps to nodes.
- Creates sampling pedestal / offset.

4. Clock feedthrough
- Clock couples through `Cgd/Cgs` into hold node.
- Injects clock-correlated glitch.

5. Hold-mode droop
- Leakage paths change held voltage over time.

6. Nonlinear switch resistance
- `RON` depends on signal level.
- Leads to distortion unless bootstrapped switch used.

---

## 5) Design Tradeoffs

Increasing `CH`:
- Pros: lower `kT/C` noise, lower droop
- Cons: slower settling, bigger drive, higher power

Reducing `RON`:
- Pros: faster acquisition
- Cons: larger switch capacitance/charge injection, clock load

Using bootstrapped switch:
- Pros: flatter `RON` vs input, better linearity
- Cons: extra circuit complexity and reliability concerns

Using buffer after hold cap:
- Pros: isolates backend kickback/load
- Cons: buffer noise/offset/power

---

## 6) Fast Exam Workflow for S/H Questions

1. Compute `LSB = VFS/2^N`
2. Convert spec to error budget (`<= 0.5 LSB`)
3. Check settling using RC exponential
4. Check `kT/C` vs LSB budget
5. Check droop over hold interval
6. If high-frequency input, check jitter SNR limit

---

## 7) Common Oral/Short Questions

1. Why is S/H necessary in SAR/pipeline ADCs?
2. Why does larger hold capacitor improve noise but hurt speed?
3. Difference between charge injection and clock feedthrough?
4. Which error worsens with higher input frequency: droop or jitter?

