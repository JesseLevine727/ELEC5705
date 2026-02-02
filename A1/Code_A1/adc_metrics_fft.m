function [SNR, SINAD, SFDR, ENOB, k0, P, PdBFS, f] = adc_metrics_fft(y, fs, M, Vpp, guard, max_harmonic)
    Y = fft(y, M);
    Knyq = M/2;

    P2 = (abs(Y)/M).^2;
    P1 = P2(1:Knyq+1);
    P = P1; P(2:end-1) = 2*P(2:end-1);

    f = (0:Knyq).'* (fs/M);

    Vrms_fs = (Vpp/2)/sqrt(2);
    PdBFS = 10*log10(P/(Vrms_fs^2) + eps);

    [~, kFund] = max(P(2:end));
    kFund = kFund + 1;
    k0 = kFund - 1;

    mark = @(mask,k) set_bins(mask,k,guard,Knyq);

    mask_dc = false(size(P));
    mask_dc(1) = true;

    mask_sig = false(size(P));
    mask_sig = mark(mask_sig, k0);
    Psig = sum(P(mask_sig));

    % SINAD
    mask_sinad_excl = false(size(P));
    mask_sinad_excl = mask_dc;
    mask_sinad_excl = mark(mask_sinad_excl, k0);
    Pnd = sum(P(~mask_sinad_excl));
    SINAD = 10*log10(Psig/(Pnd + eps));

    % SNR (exclude harmonics)
    mask_snr_excl = false(size(P));
    mask_snr_excl = mask_dc;
    mask_snr_excl = mark(mask_snr_excl, k0);

    for h = 2:max_harmonic
        kh = mod(h*k0, M);
        if kh > Knyq, kh = M - kh; end
        mask_snr_excl = mark(mask_snr_excl, kh);
    end
    Pnoise = sum(P(~mask_snr_excl));
    SNR = 10*log10(Psig/(Pnoise + eps));

    % SFDR
    mask_sfdr_excl = false(size(P));
    mask_sfdr_excl = mask_dc;
    mask_sfdr_excl = mark(mask_sfdr_excl, k0);
    Pspur = max(P(~mask_sfdr_excl));
    SFDR = 10*log10(Psig/(Pspur + eps));

    ENOB = (SINAD - 1.76)/6.02;
end
