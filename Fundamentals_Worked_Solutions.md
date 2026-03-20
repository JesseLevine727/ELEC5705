# ELEC5705 Fundamentals Worked Solutions (Companion)

Companion to:
- [Fundamentals_Study_Guide.md](/home/elfo/Documents/ELEC5705/Fundamentals_Study_Guide.md)

This document solves three high-probability fundamentals question types step by step.

---

## Problem 1: Nyquist + Aliasing

An input signal has bandwidth `B = 120 MHz`.
1. Check Nyquist for:
   - (a) `fs = 180 MS/s`
   - (b) `fs = 300 MS/s`
2. For each case, find where a tone at `fin = 170 MHz` appears in the first Nyquist zone.

## Solution

Given:
- Nyquist condition: `fs >= 2B`
- Here `2B = 240 MHz`

### 1) Nyquist check

(a) `fs = 180 MS/s`
- Compare: `180 < 240` -> Nyquist not satisfied.
- Perfect reconstruction of a 120 MHz-bandwidth signal is not guaranteed.

(b) `fs = 300 MS/s`
- Compare: `300 >= 240` -> Nyquist satisfied.
- Reconstruction is possible (assuming proper anti-alias filtering and bandlimit).

### 2) Alias of `fin = 170 MHz`

Use fast folding method:
1. Compute `fin mod fs`
2. If result `> fs/2`, fold: `f_alias = fs - result`

(a) `fs = 180 MHz`
- `fin mod fs = 170`
- `fs/2 = 90`, and `170 > 90` -> fold
- `f_alias = 180 - 170 = 10 MHz`

(b) `fs = 300 MHz`
- `fin mod fs = 170`
- `fs/2 = 150`, and `170 > 150` -> fold
- `f_alias = 300 - 170 = 130 MHz`

Final answers:
- Nyquist satisfied? (a) No, (b) Yes
- Tone location in first Nyquist zone:
  - (a) `10 MHz`
  - (b) `130 MHz`

---

## Problem 2: LSB + Quantization Noise + Ideal SNR

An ideal ADC has:
- `N = 8 bits`
- `VFS = 2.0 Vpp` (range from `-1 V` to `+1 V`)

Find:
1. `LSB`
2. Quantization noise power `sigma_q^2`
3. Quantization noise RMS `sigma_q`
4. Ideal full-scale sine SNR

## Solution

### 1) LSB

`Delta = VFS / 2^N = 2 / 256 = 0.0078125 V = 7.8125 mV`

### 2) Quantization noise power

`sigma_q^2 = Delta^2 / 12`

`Delta^2 = (0.0078125)^2 = 6.1035e-5`

`sigma_q^2 = 6.1035e-5 / 12 = 5.0863e-6 V^2`

### 3) Quantization noise RMS

`sigma_q = Delta / sqrt(12) = 0.0078125 / 3.4641 = 0.002255 V`

So:
- `sigma_q ≈ 2.255 mV rms`

### 4) Ideal SNR

`SNR_ideal(dB) = 6.02N + 1.76`

`SNR_ideal = 6.02*8 + 1.76 = 49.92 dB`

Final answers:
- `LSB = 7.8125 mV`
- `sigma_q^2 ≈ 5.09e-6 V^2`
- `sigma_q ≈ 2.255 mV rms`
- `SNR_ideal ≈ 49.92 dB`

---

## Problem 3: FFT Metrics (SINAD/SNDR, SFDR, ENOB)

From an FFT-based ADC test:
- Fundamental power: `P_signal = -1 dBFS`
- Integrated noise + distortion (excluding DC and fundamental): `P_ND = -52 dBFS`
- Largest spur: `P_spur = -60 dBFS`

Find:
1. `SINAD` (same as SNDR in many notes)
2. `SFDR` in dBc
3. `ENOB`
4. Is behavior more noise-limited or distortion-limited?

## Solution

Because all values are already in dBFS, ratio in dB is a subtraction.

### 1) SINAD

`SINAD = P_signal - P_ND = (-1) - (-52) = 51 dB`

### 2) SFDR (dBc)

`SFDR = P_signal - P_spur = (-1) - (-60) = 59 dBc`

### 3) ENOB

`ENOB = (SINAD - 1.76)/6.02 = (51 - 1.76)/6.02 = 49.24/6.02 = 8.18 bits`

### 4) Interpretation

- `SFDR = 59 dBc` is much larger than SINAD gap contributors implied by `P_ND`.
- No single spur is dominating near the carrier limit.
- This looks more noise-floor + aggregate-distortion limited than single-spur limited.

Final answers:
- `SINAD = 51 dB`
- `SFDR = 59 dBc`
- `ENOB ≈ 8.18 bits`
- Classification: mostly noise/aggregate-ND limited, not worst-spur limited.

---

## Quick Self-Check Prompts

1. If `N` increases by 1 bit, how much should ideal SNR increase?
2. If `fs` doubles while signal bandwidth stays fixed, what happens to Nyquist margin?
3. If SINAD drops but SFDR stays nearly constant, what likely got worse: noise floor or largest spur?

Expected quick answers:
1. About `+6.02 dB`
2. Margin increases (aliasing risk decreases)
3. Noise floor / distributed distortion likely worsened

