% Assignment 1 - 8-bit ADC Characterizatiom
% Part 1: Ideal 8-bit ADC
%% Generating Coherent input sine x[n]
clear;
clc;
close all;

fs = 10e9;
M = 8192;
Vpp = 2.0;
A = 0.98*(Vpp/2);
phi =0;

prime = 211;
fin = (prime/M)*fs;
n = (0:M-1).';
t = n/fs;

x = A*sin(2*pi*fin*t+phi);

fprintf('fs   = %.3f GHz\n', fs/1e9);
fprintf('M    = %d\n', M);
fprintf('prime    = %d\n', prime);
fprintf('fin  = %.6f MHz\n', fin/1e6);

Ns = 200;
figure;
plot(t(1:Ns)*1e9, x(1:Ns), 'LineWidth', 1);
grid on;
xlabel('Time (ns)'); ylabel('Voltage (V)');
title('Input sequence x[n] = A sin(2π f_{in} n / f_s)');

Ns = 200;
tseg = t(1:Ns);
tfine = linspace(tseg(1), tseg(end), 5000);
xfine = A*sin(2*pi*fin*tfine + phi);

figure;
plot(tfine*1e9, xfine, '-'); hold on;
plot(tseg*1e9, x(1:Ns), 'o');
grid on;
xlabel('Time (ns)'); ylabel('V');
title('Continuous sine (line) + sampled points (markers)');
legend('Continuous reference','Samples');

%% Digitize sampled sequence
N = 8;
LSB = Vpp / (2^N);

x_clip = min(max(x, -Vpp/2), (Vpp/2 - LSB));
code = floor( (x_clip + Vpp/2) / LSB );
code = max(0, min(code, 2^N - 1));

y = (code+0.5)*LSB-Vpp/2;
fprintf("LSB (Delta) = %.6f V\n", LSB);
fprintf("x:    min=%.6f, max=%.6f\n", min(x), max(x));
fprintf("y:    min=%.6f, max=%.6f\n", min(y), max(y));
fprintf("code: min=%d, max=%d\n", min(code), max(code));

Ns = 200;
figure;
plot(t(1:Ns)*1e9, x(1:Ns), 'LineWidth', 1); hold on;
stairs(t(1:Ns)*1e9, y(1:Ns), 'LineWidth', 1);
grid on;
xlabel('Time (ns)'); ylabel('Voltage (V)');
title('Sampled input x[n] and digitized (quantized) output y[n]');
legend('x[n] (sampled input)', 'y[n] (quantized output)');
figure;
stem(t(1:Ns)*1e9, code(1:Ns), 'filled');
grid on;
xlabel('Time (ns)'); ylabel('ADC code');
title('Digitized ADC output codes c[n]');
%% Computing SNR, SFDR, SINAD, ENOB
guard = 1;
max_harmonic = 5;
Y = fft(y,M);

Knyq = M/2;
P2 = (abs(Y)/M).^2;
P1 = P2(1:Knyq+1);

P = P1;
P(2:end-1) = 2*P(2:end-1);

f = (0:Knyq).'* (fs/M);

% --- Convert to dBFS (power) ---
Vrms_fs = (Vpp/2)/sqrt(2);   % full-scale sine RMS (for dBFS reference)
PdBFS = 10*log10(P/(Vrms_fs^2) + eps);

[~, kFund] = max(P(2:end));  
kFund = kFund + 1;           
k0 = kFund - 1;

mark = @(mask,k) set_bins(mask,k,guard,Knyq);

mask_dc = false(size(P));
mask_dc(1) = true;           % DC bin

% Signal mask
mask_sig = false(size(P));
mask_sig = mark(mask_sig, k0);

Psig = sum(P(mask_sig));

% SINAD - Noise + Distortion
mask_sinad_excl = false(size(P));
mask_sinad_excl = mask_dc;
mask_sinad_excl = mark(mask_sinad_excl, k0);

Pnd = sum(P(~mask_sinad_excl));
SINAD = 10*log10(Psig/(Pnd + eps));    % dB

%SNR 
mask_snr_excl = false(size(P));
mask_snr_excl = mask_dc;
mask_snr_excl = mark(mask_snr_excl, k0);

for h = 2:max_harmonic
    kh = mod(h*k0, M);       % 0..M-1 (0-based)
    if kh > Knyq
        kh = M - kh;         % fold to 0..Knyq (alias)
    end
    mask_snr_excl = mark(mask_snr_excl, kh);
end

Pnoise = sum(P(~mask_snr_excl));
SNR = 10*log10(Psig/(Pnoise + eps));    % dB

%SFDR - Spurious Free Dyn. Range
mask_sfdr_excl = false(size(P));
mask_sfdr_excl = mask_dc;
mask_sfdr_excl = mark(mask_sfdr_excl, k0);

Pspur = max(P(~mask_sfdr_excl));
SFDR = 10*log10(Psig/(Pspur + eps));    % dBc (relative to carrier)

%ENOB
ENOB = (SINAD - 1.76)/6.02;

fprintf('\n--- FFT Metrics (8192-pt) ---\n');
fprintf('Fundamental bin = %d (%.6f MHz)\n', k0, f(k0+1)/1e6);
fprintf('SNR   = %.2f dB\n', SNR);
fprintf('SINAD = %.2f dB\n', SINAD);
fprintf('SFDR  = %.2f dBc\n', SFDR);
fprintf('ENOB  = %.2f bits\n', ENOB);


%% ---- FFT plot with ADI-style horizontal markers + SFDR brackets ----

% Levels in dBFS (power)
carrier_dBFS = 10*log10(Psig/(Vrms_fs^2) + eps);
spur_dBFS    = 10*log10(Pspur/(Vrms_fs^2) + eps);

figure;
plot(f/1e9, PdBFS, 'LineWidth', 1);
grid on;
xlabel('Frequency (GHz)');
ylabel('Power (dBFS)');
title(sprintf('Output Spectrum (M=%d, fs=%.1f GS/s, fin=%.3f MHz)', M, fs/1e9, fin/1e6));
xlim([0 fs/(2*1e9)]);

% Give yourself some headroom for labels
ylim([-160 5]);
hold on;

% --- Horizontal markers ---
yline(0, 'k--', 'LineWidth', 1);               % Full-scale (0 dBFS)
yline(carrier_dBFS, 'k--', 'LineWidth', 1);   % Carrier level
yline(spur_dBFS, 'k--', 'LineWidth', 1);      % Worst spur level

% Labels for markers (simple, not cluttered)
ax = gca;
xr = ax.XLim;
x_textL = xr(1) + 0.02*(xr(2)-xr(1));
x_textR = xr(1) + 0.62*(xr(2)-xr(1));

text(x_textL,  0+2,           'FULL SCALE (0 dBFS)', 'FontWeight','bold');
text(x_textR,  spur_dBFS+2,   'WORST SPUR LEVEL',    'FontWeight','bold');

% --- SFDR brackets ---
% SFDR(dBc): carrier - spur
x_sfdr_dbc = xr(1) + 0.80*(xr(2)-xr(1));
draw_vbracket(x_sfdr_dbc, spur_dBFS, carrier_dBFS, sprintf('SFDR = %.2f dBc', SFDR));

% SFDR(dBFS): fullscale - spur
x_sfdr_dbfs = xr(1) + 0.90*(xr(2)-xr(1));
draw_vbracket(x_sfdr_dbfs, spur_dBFS, 0, sprintf('SFDR = %.2f dBFS', -spur_dBFS));

metrics_txt = sprintf(['SINAD = %.2f dB\n' ...
                       'SNR   = %.2f dB\n' ...
                       'SFDR  = %.2f dBc\n' ...
                       'ENOB  = %.2f bits'], ...
                       SINAD, SNR, SFDR, ENOB);

x_box = xr(1) + 0.60*(xr(2)-xr(1));
y_box = ax.YLim(1) + 0.90*(ax.YLim(2)-ax.YLim(1));
text(x_box, y_box, metrics_txt, ...
    'FontName','Consolas', 'FontSize', 11, ...
    'BackgroundColor','w', 'EdgeColor','k', 'Margin', 8);

hold off;
%% Part B - Adding Random Noise
% Find max random noise without losing more than 1-bit ENOB
% Gaussian noise is added before quantization
% Model: x_noisy = x + sigma * randn()
% Estimate ENOB via averaging over multiple noise realizations.

ENOB0    = ENOB;            % baseline ENOB from ideal case above
ENOB_min = ENOB0 - 1.0;     % allowed drop <= 1 bit

MC = 20;                    % number of noise realizations per sigma (increase if jittery)
rng(1);                     % reproducible runs

relTol  = 1e-3;             % relative tolerance on sigma bracket
maxIter = 40;

fprintf('\n--- Part B (Refined): Additive random noise ---\n');
fprintf('Baseline ENOB0 = %.4f bits; Threshold ENOB_min = %.4f bits\n', ENOB0, ENOB_min);

% Function handle: average ENOB over MC noise realizations
evalENOB = @(sigma) mean_enob_for_sigma(sigma, MC, x, Vpp, LSB, N, fs, M, guard, max_harmonic);

% -------------------- 1) Bracket the boundary --------------------
sigma_lo = 0;                     % definitely safe
ENOB_lo  = evalENOB(sigma_lo);

sigma_hi = 1e-6;                  % start very small
ENOB_hi  = evalENOB(sigma_hi);

grow = 2.0;
sigma_hi_max = 10*Vpp;            % safety cap (absurdly large)

while ENOB_hi >= ENOB_min
    sigma_lo = sigma_hi;
    ENOB_lo  = ENOB_hi;

    sigma_hi = sigma_hi * grow;
    if sigma_hi > sigma_hi_max
        error('Failed to bracket: ENOB never dropped below threshold up to sigma=%.3g', sigma_hi_max);
    end
    ENOB_hi = evalENOB(sigma_hi);
end

fprintf('Bracket found:\n');
fprintf('  sigma_lo = %.6g Vrms (ENOB=%.4f)  [SAFE]\n', sigma_lo, ENOB_lo);
fprintf('  sigma_hi = %.6g Vrms (ENOB=%.4f)  [FAIL]\n', sigma_hi, ENOB_hi);

% -------------------- 2) Binary search in log-space --------------------
for it = 1:maxIter
    sigma_mid = 10.^((log10(sigma_lo) + log10(sigma_hi))/2);
    ENOB_mid  = evalENOB(sigma_mid);

    if ENOB_mid >= ENOB_min
        sigma_lo = sigma_mid;
        ENOB_lo  = ENOB_mid;
    else
        sigma_hi = sigma_mid;
        ENOB_hi  = ENOB_mid;
    end

    if (sigma_hi/sigma_lo - 1) < relTol
        break;
    end
end

sigma_star = sigma_lo;     % max tolerated (still safe)
ENOB_star  = ENOB_lo;

fprintf('\nRefined max tolerated noise:\n');
fprintf('  sigma* = %.8g Vrms\n', sigma_star);
fprintf('  ENOB(sigma*) = %.4f bits (drop = %.4f bits)\n', ENOB_star, ENOB0-ENOB_star);
fprintf('  Final bracket ratio hi/lo - 1 = %.3g\n', (sigma_hi/sigma_lo - 1));

% --- Build ONE representative spectrum at sigma* (single realization) ---
w = randn(size(x));
x_noisy = x + sigma_star*w;

% Clip to ADC input range (avoid hard clipping dominating)
x_noisy = min(max(x_noisy, -Vpp/2), (Vpp/2 - LSB));

% Quantize
code_n = floor((x_noisy + Vpp/2)/LSB);
code_n = max(0, min(code_n, 2^N - 1));
y_n = (code_n + 0.5)*LSB - Vpp/2;

% Compute metrics for this realization (for plot)
[SNR_n, SINAD_n, SFDR_n, ENOB_n, k0_n, P_n, PdBFS_n, f_n] = ...
    adc_metrics_fft(y_n, fs, M, Vpp, guard, max_harmonic);

fprintf('\nOne-realization metrics at sigma* (for plotted spectrum):\n');
fprintf('  SNR   = %.2f dB\n', SNR_n);
fprintf('  SINAD = %.2f dB\n', SINAD_n);
fprintf('  SFDR  = %.2f dBc\n', SFDR_n);
fprintf('  ENOB  = %.2f bits\n', ENOB_n);

%% ---- Plot noisy FFT with the same ADI-style markers + SFDR brackets ----
Knyq = M/2;
mask_sfdr_excl = false(size(P_n));
mask_sfdr_excl(1) = true;                  % DC
mask_sfdr_excl = set_bins(mask_sfdr_excl, k0_n, guard, Knyq);

Psig_n = sum(P_n(set_bins(false(size(P_n)), k0_n, guard, Knyq)));
Pspur_n = max(P_n(~mask_sfdr_excl));

Vrms_fs = (Vpp/2)/sqrt(2);
carrier_dBFS = 10*log10(Psig_n/(Vrms_fs^2) + eps);
spur_dBFS    = 10*log10(Pspur_n/(Vrms_fs^2) + eps);

figure;
plot(f_n/1e9, PdBFS_n, 'LineWidth', 1);
grid on; xlabel('Frequency (GHz)'); ylabel('Power (dBFS)');
title(sprintf('Noisy Output Spectrum (ENOB drop target 1 bit, \\sigma*=%.3g Vrms)', sigma_star));
xlim([0 fs/(2*1e9)]); ylim([-160 5]); hold on;

yline(0,'k--','LineWidth',1);
yline(carrier_dBFS,'k--','LineWidth',1);
yline(spur_dBFS,'k--','LineWidth',1);

ax = gca; xr = ax.XLim;
x_textL = xr(1)+0.02*(xr(2)-xr(1));
x_textR = xr(1)+0.62*(xr(2)-xr(1));
text(x_textR, spur_dBFS+2, 'WORST SPUR LEVEL','FontWeight','bold');

x_sfdr_dbc  = xr(1)+0.80*(xr(2)-xr(1));
x_sfdr_dbfs = xr(1)+0.90*(xr(2)-xr(1));
draw_vbracket(x_sfdr_dbc,  spur_dBFS, carrier_dBFS, sprintf('SFDR = %.2f dBc', SFDR_n));
draw_vbracket(x_sfdr_dbfs, spur_dBFS, 0,           sprintf('SFDR = %.2f dBFS', -spur_dBFS));

metrics_txt = sprintf(['SINAD = %.2f dB\n' ...
                       'SNR   = %.2f dB\n' ...
                       'SFDR  = %.2f dBc\n' ...
                       'ENOB  = %.2f bits\n' ...
                       '\\sigma* = %.3g Vrms'], ...
                       SINAD_n, SNR_n, SFDR_n, ENOB_n, sigma_star);

x_box = xr(1)+0.60*(xr(2)-xr(1));
y_box = ax.YLim(1)+0.90*(ax.YLim(2)-ax.YLim(1));
text(x_box, y_box, metrics_txt, ...
    'FontName','Consolas','FontSize',11,'BackgroundColor','w','EdgeColor','k','Margin',8);

hold off;
