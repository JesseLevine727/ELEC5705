function ENOB = enob_for_a3(a3, x, Vpp, LSB, N, fs, M, guard, max_harmonic)
    y_nl = quantize_nl(x, a3, Vpp, LSB, N);
    [~, ~, ~, ENOB, ~, ~, ~, ~] = adc_metrics_fft(y_nl, fs, M, Vpp, guard, max_harmonic);
end