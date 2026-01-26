function ENOBm = mean_enob_for_sigma(sigma, MC, x, Vpp, LSB, N, fs, M, guard, max_harmonic)
% Average ENOB over MC random noise realizations for additive white Gaussian noise.
    ENOBs = zeros(MC,1);
    for i = 1:MC
        w = randn(size(x));
        x_noisy = x + sigma*w;

        % keep within ADC input range (avoid hard clipping dominating)
        x_noisy = min(max(x_noisy, -Vpp/2), (Vpp/2 - LSB));

        % quantize
        code = floor((x_noisy + Vpp/2)/LSB);
        code = max(0, min(code, 2^N - 1));
        y = (code + 0.5)*LSB - Vpp/2;

        [~, ~, ~, ENOBs(i)] = adc_metrics_fft(y, fs, M, Vpp, guard, max_harmonic);
    end
    ENOBm = mean(ENOBs);
end
