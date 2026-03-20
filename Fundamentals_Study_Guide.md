# ELEC5705 Fundamentals Study Guide (20/80)

This guide focuses on the small set of concepts and equations that usually generate most marks in "fundamentals" questions.

## 1) The Core Mental Model

An ADC does 3 jobs:
1. `Sample` in time (`fs`)
2. `Quantize` in amplitude (`N` bits, `2^N` levels)
3. `Encode` to digital code

Most fundamentals questions reduce to:
1. Is sampling fast enough? (Nyquist/aliasing)
2. Is quantization fine enough? (LSB/noise/SNR)
3. How good is real dynamic performance? (SINAD/SNDR, SFDR, ENOB)

---

## 2) Sampling: What You Must Know

## 2.1 Nyquist condition
- If input bandwidth is `B`, then minimum sampling rate:
- `fs >= 2B`
- Nyquist frequency is `fN = fs/2`.

Interpretation:
- Content above `fs/2` folds (aliases) into baseband.
- If exam asks "can you reconstruct perfectly?", check if signal is bandlimited below `fs/2`.

## 2.2 Aliasing frequency map
Aliased components appear at:
- `f_alias = |±Kfs ± fin|` for integer `K`
- Then fold into `[0, fs/2]`.

Fast method in problems:
1. Compute `fin mod fs`
2. If result `> fs/2`, fold with `fs - result`.

---

## 3) Quantization: The Main Equations

## 3.1 LSB and code spacing
- Full-scale range: `VFS`
- Resolution: `N` bits
- `LSB = Delta = VFS / 2^N`

## 3.2 Quantization error model (ideal)
- `e_q = x - x_q`
- For ideal uniform quantizer:
- `e_q in [-Delta/2, +Delta/2]`

Assuming uncorrelated/uniform error:
- Quantization noise power: `sigma_q^2 = Delta^2 / 12`
- Quantization noise RMS: `sigma_q = Delta / sqrt(12)`

## 3.3 Ideal SNR (full-scale sine)
- `SNR_ideal(dB) = 6.02N + 1.76`

This single equation is one of the highest-yield equations in the course.

---

## 4) FFT-Based Dynamic Metrics (Most Tested)

In labs/homework and many exam questions, ADC performance is extracted from FFT bins.

## 4.1 Frequency-bin basics
- FFT length `M` -> bin spacing:
- `Delta f = fs / M`
- Coherent test tone:
- `fin = (k/M) * fs` where `k` is integer (often chosen relatively prime to `M`)

Reason:
- Coherent sampling reduces leakage and makes metric extraction cleaner.

## 4.2 Definitions (know exactly)

`SNR`:
- `SNR = 10log10(P_signal / P_noise)`
- Harmonics are excluded from noise for SNR.

`SINAD` (or `SNDR` in many notes):
- `SINAD = 10log10(P_signal / (P_noise + P_distortion))`
- Includes harmonics and noise (except DC and fundamental).

`SFDR`:
- `SFDR = 10log10(P_signal / P_largest_spur)`
- Largest non-fundamental tone only.

`ENOB` from SINAD:
- `ENOB = (SINAD - 1.76)/6.02`

Priority relation you should remember:
- Usually `SNR >= SINAD`
- If distortion is tiny, `SNR` and `SINAD` are close.

## 4.3 dB conventions
- `dBc`: relative to carrier (fundamental)
- `dBFS`: relative to full-scale

Always label your final answers with the correct reference.

---

## 5) Distortion vs Noise (Exam Interpretation)

Noise-limited ADC:
- Spectrum floor rises, harmonics may stay small.
- SNR and SINAD degrade similarly.

Distortion-limited ADC:
- Discrete harmonics/spurs dominate.
- SFDR and SINAD drop, while SNR may remain relatively better (if harmonics excluded).

What causes distortion in fundamentals context:
- Front-end nonlinearity (`x + a2 x^2 + a3 x^3 + ...`)
- Comparator kickback/nonlinearity
- CDAC mismatch (architecture-dependent)

---

## 6) Oversampling (High-Yield Shortcut)

Oversampling ratio:
- `OSR = fs / (2B)`

For quantization-noise-limited case (no shaping):
- Doubling `OSR` improves SNR by about `3 dB`.

This is often asked conceptually, even before delta-sigma details.

---

## 7) Fast Exam Workflow (Use This Every Time)

1. List knowns: `fs, fin, B, N, VFS, M`
2. Check Nyquist (`fs ? 2B`) and alias location
3. Compute `LSB`, and ideal `SNR_ideal = 6.02N + 1.76`
4. If FFT powers are given, compute:
   - `SINAD`
   - `SFDR`
   - `ENOB`
5. State units clearly (`dB`, `dBc`, `dBFS`, bits)
6. Add one interpretation line:
   - "noise-limited" or "distortion-limited"

---

## 8) Minimal Equation Sheet (Memorize)

- `fs >= 2B`
- `fN = fs/2`
- `f_alias = |±Kfs ± fin|`
- `Delta = VFS/2^N`
- `sigma_q^2 = Delta^2/12`
- `SNR_ideal = 6.02N + 1.76`
- `Delta f = fs/M`
- `SINAD = 10log10(Ps/(Pn + Pd))`
- `SFDR = 10log10(Ps/Pspur_max)`
- `ENOB = (SINAD - 1.76)/6.02`
- `OSR = fs/(2B)`

If you can derive and interpret these confidently, you have the 20% that drives most fundamentals marks.

---

## 9) Three Likely Fundamentals Questions

1. Given `fs`, `B`, and a tone `fin`, determine whether Nyquist is satisfied and find aliased output frequency.
2. For an `N`-bit ADC with `VFS`, compute `LSB`, quantization noise power, and ideal SNR.
3. From an FFT table (fundamental, total noise+distortion, largest spur), compute `SINAD/SNDR`, `SFDR`, and `ENOB`, then classify noise-limited vs distortion-limited.

