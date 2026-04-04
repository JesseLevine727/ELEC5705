import math
import numpy as np

COMP_BASE_ERR = 20
DAC_BASE_ERRS = [20, 18, 16, 14, 12]


def bit_weight(idx, trims, dac_scale=(1.0 / (8192.0 * 16.0))):
    mismatch = 0.0
    if idx == 14:
        ideal = 1.0 / 2.0
        mismatch = -(DAC_BASE_ERRS[0] - trims[0]) * dac_scale
    elif idx == 13:
        ideal = 1.0 / 4.0
        mismatch = -(DAC_BASE_ERRS[1] - trims[1]) * dac_scale
    elif idx == 12:
        ideal = 1.0 / 8.0
        mismatch = -(DAC_BASE_ERRS[2] - trims[2]) * dac_scale
    elif idx == 11:
        ideal = 1.0 / 16.0
        mismatch = -(DAC_BASE_ERRS[3] - trims[3]) * dac_scale
    elif idx == 10:
        ideal = 1.0 / 32.0
        mismatch = -(DAC_BASE_ERRS[4] - trims[4]) * dac_scale
    elif idx == 9:
        ideal = 1.0 / 64.0
    elif idx == 8:
        ideal = 1.0 / 128.0
    elif idx == 7:
        ideal = 1.0 / 256.0
    elif idx == 6:
        ideal = 1.0 / 512.0
    elif idx == 5:
        ideal = 1.0 / 1024.0
    elif idx == 4:
        ideal = 1.0 / 2048.0
    elif idx == 3:
        ideal = 1.0 / 4096.0
    elif idx == 2:
        ideal = 1.0 / 4096.0
    elif idx == 1:
        ideal = 1.0 / 8192.0
    elif idx == 0:
        ideal = 1.0 / 8192.0
    else:
        ideal = 0.0
    return ideal + mismatch


def build_codebook(trims, dac_scale=(1.0 / (8192.0 * 16.0))):
    w = np.array([bit_weight(i, trims, dac_scale) for i in range(14, -1, -1)])
    codes = np.arange(8192, dtype=np.uint16)
    bits = ((codes[:, None] >> np.arange(12, -1, -1)) & 1).astype(float)
    # map 13-bit code into weights 14..2, ignoring redundant tail bits
    used_weights = w[:13]
    vdac = bits @ used_weights
    return vdac


def quantize(vin, vdac, comp_trim, comp_scale=(1.0 / (8192.0 * 4.0))):
    delta_off = -(COMP_BASE_ERR - comp_trim) * comp_scale
    target = vin + delta_off
    idx = np.searchsorted(vdac, target, side="right") - 1
    idx = np.clip(idx, 0, len(vdac) - 1)
    return idx


def sndr_db(codes):
    x = codes.astype(float)
    x = x - np.mean(x)
    n = len(x)
    win = np.hanning(n)
    xw = x * win
    spec = np.fft.rfft(xw)
    mag2 = np.abs(spec) ** 2
    mag2[0] = 0.0
    fund = np.argmax(mag2[1:]) + 1
    signal = mag2[fund]
    noise_dist = np.sum(mag2) - signal
    return 10.0 * math.log10(signal / noise_dist)


def evaluate(name, comp_trim, dac_trims, dac_scale):
    vdac = build_codebook(dac_trims, dac_scale)
    n = 8192
    k = 37
    t = np.arange(n)
    vin = 0.5 + 0.49 * np.sin(2 * math.pi * k * t / n)
    codes = quantize(vin, vdac, comp_trim)
    s = sndr_db(codes)
    print(f"{name:12s} dac_scale={dac_scale:.6g} SNDR={s:.3f} dB ENOB={(s-1.76)/6.02:.3f}")


if __name__ == "__main__":
    scenarios = [
        ("uncal", 0, [0, 0, 0, 0, 0]),
        ("comp_only", 20, [0, 0, 0, 0, 0]),
        ("both", 20, [20, 18, 16, 14, 12]),
    ]
    for scale in [1.0 / (8192.0 * 16.0), 1.0 / (8192.0 * 8.0), 1.0 / (8192.0 * 4.0), 1.0 / (8192.0 * 2.0)]:
        print(f"\nscale={scale}")
        for name, ct, dt in scenarios:
            evaluate(name, ct, dt, scale)
