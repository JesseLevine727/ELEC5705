#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import numpy as np

COMP_BASE_ERR = 20
DAC_BASE_ERRS = [20, 18, 16, 14, 12]
COMP_SCALE = 1.0 / (8192.0 * 6.0)
DAC_SCALE = 1.0 / (8192.0 * 16.0)
FS_MHZ = 6.4
NFFT = 8192
FIN_BIN = 37


def bit_weight(idx, trims):
    mismatch = 0.0
    if idx == 14:
        ideal = 1.0 / 2.0
        mismatch = -(DAC_BASE_ERRS[0] - trims[0]) * DAC_SCALE
    elif idx == 13:
        ideal = 1.0 / 4.0
        mismatch = -(DAC_BASE_ERRS[1] - trims[1]) * DAC_SCALE
    elif idx == 12:
        ideal = 1.0 / 8.0
        mismatch = -(DAC_BASE_ERRS[2] - trims[2]) * DAC_SCALE
    elif idx == 11:
        ideal = 1.0 / 16.0
        mismatch = -(DAC_BASE_ERRS[3] - trims[3]) * DAC_SCALE
    elif idx == 10:
        ideal = 1.0 / 32.0
        mismatch = -(DAC_BASE_ERRS[4] - trims[4]) * DAC_SCALE
    else:
        ideal = {
            9: 1.0 / 64.0,
            8: 1.0 / 128.0,
            7: 1.0 / 256.0,
            6: 1.0 / 512.0,
            5: 1.0 / 1024.0,
            4: 1.0 / 2048.0,
            3: 1.0 / 4096.0,
            2: 1.0 / 4096.0,
            1: 1.0 / 8192.0,
            0: 1.0 / 8192.0,
        }.get(idx, 0.0)
    return ideal + mismatch


def build_vdac(trims):
    weights = np.array([bit_weight(i, trims) for i in range(14, -1, -1)])[:13]
    codes = np.arange(8192, dtype=np.uint16)
    bits = ((codes[:, None] >> np.arange(12, -1, -1)) & 1).astype(float)
    return bits @ weights


def evaluate_codes(comp_trim, dac_trims):
    t = np.arange(NFFT)
    vin = 0.5 + 0.49 * np.sin(2.0 * math.pi * FIN_BIN * t / NFFT)
    vdac = build_vdac(dac_trims)
    coarse = np.searchsorted(vdac, vin, side="right") - 1
    coarse = np.clip(coarse, 0, len(vdac) - 1)
    residue_lsb = (vin - vdac[coarse]) * 8192.0
    target = vin - (COMP_BASE_ERR - comp_trim) * COMP_SCALE * (1.0 + residue_lsb)
    codes = np.searchsorted(vdac, target, side="right") - 1
    return np.clip(codes, 0, len(vdac) - 1)


def spectrum_and_sndr(codes):
    x = codes.astype(float) - np.mean(codes)
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    mag2 = mag ** 2
    mag2[0] = 0.0
    fund = np.argmax(mag2[1:]) + 1
    signal = mag2[fund]
    noise_dist = np.sum(mag2) - signal
    sndr_db = 10.0 * math.log10(signal / noise_dist)
    mag_norm = mag / max(mag[fund], 1e-20)
    dbfs = 20.0 * np.log10(np.maximum(mag_norm, 1e-12))
    freqs_mhz = np.arange(len(dbfs)) * (FS_MHZ / NFFT)
    return sndr_db, freqs_mhz, dbfs


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def load_trim_log(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def generate_trim_downsample(rows, out_path):
    out_rows = []
    for idx, row in enumerate(rows):
        cycle = int(row["cycle"])
        if (cycle % 2000 == 0) or (idx == len(rows) - 1):
            out_rows.append(
                [
                    cycle,
                    int(row["trim5"]),
                    int(row["trim0"]),
                    int(row["trim1"]),
                    int(row["trim2"]),
                    int(row["trim3"]),
                    int(row["trim4"]),
                ]
            )
    write_csv(out_path, ["cycle", "trim0", "trim1", "trim2", "trim3", "trim4", "trim5"], out_rows)


def generate_sndr_trace(rows, out_path):
    out_rows = []
    for idx, row in enumerate(rows):
        cycle = int(row["cycle"])
        if (cycle % 2000 == 0) or (idx == len(rows) - 1):
            trims = [int(row[f"trim{i}"]) for i in range(5)]
            comp_trim = int(row["trim5"])
            sndr_db, _, _ = spectrum_and_sndr(evaluate_codes(comp_trim, trims))
            out_rows.append([cycle, round(sndr_db, 3)])
    write_csv(out_path, ["cycle", "sndr_db"], out_rows)


def generate_scenario_spectra(data_dir):
    scenarios = [
        ("uncal", 0, [0, 0, 0, 0, 0]),
        ("comp_only", 20, [0, 0, 0, 0, 0]),
        ("both", 20, [20, 18, 16, 14, 12]),
    ]
    summary_rows = []
    for name, comp_trim, dac_trims in scenarios:
        sndr_db, freqs_mhz, dbfs = spectrum_and_sndr(evaluate_codes(comp_trim, dac_trims))
        rows = [[round(f, 6), round(m, 3)] for f, m in zip(freqs_mhz[1:], dbfs[1:])]
        write_csv(data_dir / f"spectrum_{name}.csv", ["freq_mhz", "dbfs"], rows)
        summary_rows.append([name, round(sndr_db, 3), comp_trim, " ".join(str(v) for v in dac_trims)])
    write_csv(data_dir / "spectrum_summary.csv", ["scenario", "sndr_db", "comp_trim", "dac_trims"], summary_rows)


def main():
    report_dir = Path(__file__).resolve().parent
    project_dir = report_dir.parent
    data_dir = report_dir / "data"
    log_path = project_dir / "results" / "background_calibration_log.csv"
    rows = load_trim_log(log_path)
    generate_trim_downsample(rows, data_dir / "background_trim_downsampled.csv")
    generate_sndr_trace(rows, data_dir / "sndr_vs_cycles.csv")
    generate_scenario_spectra(data_dir)


if __name__ == "__main__":
    main()
