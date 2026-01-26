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
%% Part A - Adding Front-End Non-linearity
% Max non-linearity front-end can tolerate before losing > 1.5-bit ENOB
% Identifying the nonlinear tones in the FFT

% Approach:
% Model front-end nonlinearity before quantization:
% x_nl = x + a2*x^2 + a3*x^3
%
% Goal: find max nonlinearity that causes <= 1.5-bit ENOB degradation.
% Also: identify nonlinear tones (2nd,3rd,...) in FFT.

ENOB0    = ENOB;
ENOB_min = ENOB0 - 1.5;

relTol  = 1e-3;   % relative tolerance on a3 bracket (0.1%); tighten if you want
maxIter = 40;

fprintf('\n--- Part A (Refined): Cubic nonlinearity ---\n');
fprintf('Baseline ENOB0 = %.4f bits; Threshold ENOB_min = %.4f bits\n', ENOB0, ENOB_min);

% --- function handle to evaluate ENOB for a given a3 ---
evalENOB = @(a3) enob_for_a3(a3, x, Vpp, LSB, N, fs, M, guard, max_harmonic);

% -------------------- 1) Bracket the boundary --------------------
a3_lo = 0;                 % definitely safe
ENOB_lo = evalENOB(a3_lo);

a3_hi = 1e-6;              % start small
ENOB_hi = evalENOB(a3_hi);

% Grow a3_hi until it FAILS (ENOB < ENOB_min), keeping the last safe as lo
grow = 2.0;
a3_hi_max = 1e3;           % safety cap

while ENOB_hi >= ENOB_min
    a3_lo  = a3_hi;
    ENOB_lo = ENOB_hi;

    a3_hi = a3_hi * grow;
    if a3_hi > a3_hi_max
        error('Failed to bracket: ENOB never dropped below threshold up to a3=%.3g', a3_hi_max);
    end
    ENOB_hi = evalENOB(a3_hi);
end

fprintf('Bracket found:\n');
fprintf('  a3_lo = %.6g  (ENOB=%.4f)  [SAFE]\n', a3_lo, ENOB_lo);
fprintf('  a3_hi = %.6g  (ENOB=%.4f)  [FAIL]\n', a3_hi, ENOB_hi);

% -------------------- 2) Log-space binary search --------------------
for it = 1:maxIter
    a3_mid = 10.^((log10(a3_lo) + log10(a3_hi))/2);
    ENOB_mid = evalENOB(a3_mid);

    if ENOB_mid >= ENOB_min
        a3_lo  = a3_mid;
        ENOB_lo = ENOB_mid;
    else
        a3_hi  = a3_mid;
        ENOB_hi = ENOB_mid;
    end

    if (a3_hi/a3_lo - 1) < relTol
        break;
    end
end

a3_star = a3_lo;           % max tolerated (still safe)
ENOB_star = ENOB_lo;

fprintf('\nRefined max tolerated a3:\n');
fprintf('  a3* = %.8g  (1/V^2)\n', a3_star);
fprintf('  ENOB(a3*) = %.4f bits  (drop = %.4f bits)\n', ENOB_star, ENOB0-ENOB_star);
fprintf('  Final bracket ratio hi/lo - 1 = %.3g\n', (a3_hi/a3_lo - 1));

% If you want: compute full metrics at a3_star (SNR/SINAD/SFDR too)
[SNR_nl, SINAD_nl, SFDR_nl, ENOB_nl, k0_nl, P_nl, PdBFS_nl, f_nl] = ...
    adc_metrics_fft( quantize_nl(x, a3_star, Vpp, LSB, N), fs, M, Vpp, guard, max_harmonic );

fprintf('\nMetrics at a3*:\n');
fprintf('  SNR   = %.2f dB\n', SNR_nl);
fprintf('  SINAD = %.2f dB\n', SINAD_nl);
fprintf('  SFDR  = %.2f dBc\n', SFDR_nl);
fprintf('  ENOB  = %.2f bits\n', ENOB_nl);

% Identify main nonlinear tones (harmonics) in dBc
Vrms_fs = (Vpp/2)/sqrt(2);
carrier_mask = set_bins(false(size(P_nl)), k0_nl, guard, M/2);
Psig_nl = sum(P_nl(carrier_mask));
carrier_dBFS = 10*log10(Psig_nl/(Vrms_fs^2) + eps);

fprintf('\nNonlinear tones (harmonics) at a3*:\n');
for h = 2:5
    kh = mod(h*k0_nl, M);
    if kh > M/2, kh = M - kh; end
    tone_mask = set_bins(false(size(P_nl)), kh, guard, M/2);
    Ptone = sum(P_nl(tone_mask));
    tone_dBFS = 10*log10(Ptone/(Vrms_fs^2) + eps);
    tone_dBc  = tone_dBFS - carrier_dBFS;
    fprintf('  H%d: bin %d (%.3f MHz), level = %.2f dBc\n', h, kh, f_nl(kh+1)/1e6, tone_dBc);
end
Knyq = M/2;

mask_sfdr_excl_nl = false(size(P_nl));
mask_sfdr_excl_nl(1) = true; % DC
mask_sfdr_excl_nl = set_bins(mask_sfdr_excl_nl, k0_nl, guard, Knyq);

Pspur_nl = max(P_nl(~mask_sfdr_excl_nl));

% Levels in dBFS
Vrms_fs = (Vpp/2)/sqrt(2);
carrier_dBFS = 10*log10(Psig_nl/(Vrms_fs^2) + eps);
spur_dBFS    = 10*log10(Pspur_nl/(Vrms_fs^2) + eps);

figure;
plot(f_nl/1e9, PdBFS_nl, 'LineWidth', 1);
grid on; xlabel('Frequency (GHz)'); ylabel('Power (dBFS)');
title(sprintf('Nonlinear Output Spectrum (ENOB drop %.2f bits)', ENOB0-ENOB_nl));
xlim([0 fs/(2*1e9)]); ylim([-160 5]); hold on;

% Horizontal markers
yline(0,'k--','LineWidth',1);                 % Full scale
yline(carrier_dBFS,'k--','LineWidth',1);      % Carrier
yline(spur_dBFS,'k--','LineWidth',1);         % Worst spur

ax = gca; xr = ax.XLim;

x_textL = xr(1)+0.02*(xr(2)-xr(1));
x_textR = xr(1)+0.62*(xr(2)-xr(1));
text(x_textR, spur_dBFS+2, 'WORST SPUR LEVEL','FontWeight','bold');

% SFDR brackets
x_sfdr_dbc  = xr(1)+0.80*(xr(2)-xr(1));
x_sfdr_dbfs = xr(1)+0.90*(xr(2)-xr(1));
draw_vbracket(x_sfdr_dbc,  spur_dBFS, carrier_dBFS, sprintf('SFDR = %.2f dBc', SFDR_nl));
draw_vbracket(x_sfdr_dbfs, spur_dBFS, 0,           sprintf('SFDR = %.2f dBFS', -spur_dBFS));

% Metrics box
metrics_txt = sprintf(['SINAD = %.2f dB\n' ...
                       'SNR   = %.2f dB\n' ...
                       'SFDR  = %.2f dBc\n' ...
                       'ENOB  = %.2f bits\n' ...
                       'a_3*  = %.4g 1/V^2'], ...
                       SINAD_nl, SNR_nl, SFDR_nl, ENOB_nl, a3_star);

x_box = xr(1)+0.60*(xr(2)-xr(1));
y_box = ax.YLim(1)+0.90*(ax.YLim(2)-ax.YLim(1));
text(x_box, y_box, metrics_txt, ...
    'FontName','Consolas','FontSize',11, ...
    'BackgroundColor','w','EdgeColor','k','Margin',8);

hold off;



