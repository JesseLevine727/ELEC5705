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
%% Part C - Adding jitter
% How does Jitter impact performance?
rng(2);

sigma_t_list = [0 50 100 200 500 1000 2000 2500 3500 4500]*1e-15;  % s
Knyq = M/2;

% Preallocate results
SNR_list   = zeros(numel(sigma_t_list),1);
SINAD_list = zeros(numel(sigma_t_list),1);
SFDR_list  = zeros(numel(sigma_t_list),1);
ENOB_list  = zeros(numel(sigma_t_list),1);
SNRj_th    = zeros(numel(sigma_t_list),1);

fprintf('\n--- Part C: Sampling Jitter Sweep (FFT per point) ---\n');
fprintf('fin = %.6f MHz\n\n', fin/1e6);

for ii = 1:numel(sigma_t_list)
    sigma_t = sigma_t_list(ii);

    % Jittered sampling instants
    dt  = sigma_t * randn(size(t));
    t_j = t + dt;

    % Jittered sampled input
    x_j = A * sin(2*pi*fin*t_j + phi);

    % Quantize
    x_clip_j = min(max(x_j, -Vpp/2), (Vpp/2 - LSB));
    code_j = floor((x_clip_j + Vpp/2) / LSB);
    code_j = max(0, min(code_j, 2^N - 1));
    y_j = (code_j + 0.5)*LSB - Vpp/2;

    % Metrics + spectrum
    [SNR_j, SINAD_j, SFDR_j, ENOB_j, k0_j, P_j, PdBFS_j, f_j] = ...
        adc_metrics_fft(y_j, fs, M, Vpp, guard, max_harmonic);

    SNR_list(ii)   = SNR_j;
    SINAD_list(ii) = SINAD_j;
    SFDR_list(ii)  = SFDR_j;
    ENOB_list(ii)  = ENOB_j;

    if sigma_t == 0
        SNRj_th(ii) = Inf;
    else
        SNRj_th(ii) = -20*log10(2*pi*fin*sigma_t);
    end

    fprintf('sigma_t = %7.0f fs | SNR = %6.2f dB | SINAD = %6.2f dB | SFDR = %6.2f dBc | ENOB = %5.2f bits | SNRj(th) = %6.2f dB\n', ...
        sigma_t*1e15, SNR_j, SINAD_j, SFDR_j, ENOB_j, SNRj_th(ii));

    % ---- Build ADI-style markers for this spectrum ----
    sig_mask = set_bins(false(size(P_j)), k0_j, guard, Knyq);
    Psig_j = sum(P_j(sig_mask));

    mask_sfdr_excl = false(size(P_j));
    mask_sfdr_excl(1) = true; % DC
    mask_sfdr_excl = set_bins(mask_sfdr_excl, k0_j, guard, Knyq);
    Pspur_j = max(P_j(~mask_sfdr_excl));

    Vrms_fs = (Vpp/2)/sqrt(2);
    carrier_dBFS = 10*log10(Psig_j/(Vrms_fs^2) + eps);
    spur_dBFS    = 10*log10(Pspur_j/(Vrms_fs^2) + eps);

    % ---- FFT plot (ONE PER sigma_t) ----
    figure('Name', sprintf('FFT sigma_t = %.0f fs', sigma_t*1e15), 'NumberTitle','off');
    plot(f_j/1e9, PdBFS_j, 'LineWidth', 1);
    grid on; xlabel('Frequency (GHz)'); ylabel('Power (dBFS)');
    title(sprintf('Jittered Output Spectrum (\\sigma_t = %.0f fs RMS)', sigma_t*1e15));
    xlim([0 fs/(2*1e9)]); ylim([-160 5]); hold on;

    % Horizontal markers
    yline(0,'k--','LineWidth',1);
    yline(carrier_dBFS,'k--','LineWidth',1);
    yline(spur_dBFS,'k--','LineWidth',1);

    ax = gca; xr = ax.XLim;
    x_textL = xr(1)+0.02*(xr(2)-xr(1));
    x_textR = xr(1)+0.62*(xr(2)-xr(1));

    text(x_textL, 2, 'FULL SCALE (0 dBFS)','FontWeight','bold');
    text(x_textL, carrier_dBFS+2, 'CARRIER LEVEL','FontWeight','bold');
    text(x_textR, spur_dBFS+2, 'WORST SPUR LEVEL','FontWeight','bold');

    % SFDR brackets
    x_sfdr_dbc  = xr(1)+0.80*(xr(2)-xr(1));
    x_sfdr_dbfs = xr(1)+0.90*(xr(2)-xr(1));
    draw_vbracket(x_sfdr_dbc,  spur_dBFS, carrier_dBFS, sprintf('SFDR = %.2f dBc', SFDR_j));
    draw_vbracket(x_sfdr_dbfs, spur_dBFS, 0,           sprintf('SFDR = %.2f dBFS', -spur_dBFS));

    % Metrics box
    if isfinite(SNRj_th(ii))
        theory_line = sprintf('SNRj(th)=%.2f dB', SNRj_th(ii));
    else
        theory_line = 'SNRj(th)=Inf';
    end

    metrics_txt = sprintf(['SINAD = %.2f dB\n' ...
                           'SNR   = %.2f dB\n' ...
                           'SFDR  = %.2f dBc\n' ...
                           'ENOB  = %.2f bits\n' ...
                           '\\sigma_t = %.0f fs RMS\n' ...
                           '%s'], ...
                           SINAD_j, SNR_j, SFDR_j, ENOB_j, sigma_t*1e15, theory_line);

    x_box = xr(1)+0.60*(xr(2)-xr(1));
    y_box = ax.YLim(1)+0.90*(ax.YLim(2)-ax.YLim(1));
    text(x_box, y_box, metrics_txt, ...
        'FontName','Consolas','FontSize',11,'BackgroundColor','w','EdgeColor','k','Margin',8);

    hold off;
end

% Optional summary plots (keep if you want)
sig_fs = sigma_t_list*1e15;

figure; semilogx(sig_fs(sig_fs>0), SINAD_list(sig_fs>0), 'o-','LineWidth',1);
grid on; xlabel('\sigma_t (fs RMS)'); ylabel('SINAD (dB)'); title('SINAD vs Sampling Jitter');

figure; semilogx(sig_fs(sig_fs>0), ENOB_list(sig_fs>0), 'o-','LineWidth',1);
grid on; xlabel('\sigma_t (fs RMS)'); ylabel('ENOB (bits)'); title('ENOB vs Sampling Jitter');

figure; semilogx(sig_fs(sig_fs>0), SNR_list(sig_fs>0), 'o-','LineWidth',1);
grid on; xlabel('\sigma_t (fs RMS)'); ylabel('SNR (dB)'); title('SNR vs Sampling Jitter');

figure; semilogx(sig_fs(sig_fs>0), SFDR_list(sig_fs>0), 'o-','LineWidth',1);
grid on; xlabel('\sigma_t (fs RMS)'); ylabel('SFDR (dBc)'); title('SFDR vs Sampling Jitter');
