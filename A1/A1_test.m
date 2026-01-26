%% Assignment 1 - Part (Base): 10 GS/s, 8-bit ADC test with 8192-pt FFT
% Computes SNR, SFDR, SINAD, ENOB from FFT (per ADI MT-003 style).
clear; clc; close all;

%% ---------------- User parameters ----------------
fs  = 10e9;        % 10 GS/s
N   = 8;           % 8-bit ADC
M   = 8192;        % FFT length / record length

VFS = 2.0;         % Full-scale peak-to-peak in volts (±1 V assumed if VFS=2)
Ain = 0.99*(VFS/2);% Input sine peak amplitude slightly below full-scale (avoid clip)

% Coherent sampling: choose integer bin k relatively prime to M
% (Example like lecture figure uses k=779 for M=8192; gcd(779,8192)=1)
k = 3;
fin = (k/M)*fs;    % coherent input frequency

% Harmonics to exclude for SNR calculation (ADI commonly excludes first 5)
max_harm = 5;

% Small "bin spread" guard (sum these bins around each tone to be robust)
guard = 1;         % +/- guard bins around each identified tone bin

%% ---------------- Generate sampled input ----------------
n  = (0:M-1).';        % sample index
t  = n/fs;             % time vector
x  = Ain * sin(2*pi*fin*t);   % sampled input sine

%% ---------------- Ideal uniform quantizer (mid-tread) ----------------
% Step size (LSB) for an N-bit ADC over VFS p-p
Delta = VFS / (2^N);          % 1 LSB in volts

% Clip to ADC input range [-VFS/2, +VFS/2)
x_clip = min(max(x, -VFS/2), (VFS/2 - Delta));

% Map to code bins: code in [0, 2^N - 1]
code = floor( (x_clip + VFS/2) / Delta );
code = max(0, min(code, 2^N - 1));

% Reconstruct quantized output voltage at code centers
y = (code + 0.5)*Delta - VFS/2;

%% ---------------- FFT ----------------
% With coherent sampling, a rectangular window is OK (no leakage).
% If you later window (e.g., Hann), you must correct for coherent gain in power calcs.
Y = fft(y, M);

% Single-sided bins: 0..fs/2
Knyq = M/2;                 % Nyquist bin index in "0-based" units
bins = (0:Knyq).';          % 0..M/2
f    = bins*(fs/M);

% Power spectrum from FFT (Parseval-consistent scaling)
% Using full FFT: sum(|Y|^2)/M^2 corresponds to mean-square in time domain.
P2 = (abs(Y)/M).^2;         % two-sided power per bin (relative to volts^2)
P1 = P2(1:Knyq+1);          % single-sided slice (includes DC and Nyquist)

% Convert to single-sided power properly (double non-DC, non-Nyquist bins)
P1_ss = P1;
P1_ss(2:end-1) = 2*P1_ss(2:end-1);

% For plotting in dBFS: define full-scale sine RMS = (VFS/2)/sqrt(2)
Vrms_fs = (VFS/2)/sqrt(2);
mag_dbfs = 10*log10(P1_ss / (Vrms_fs^2) + eps);

%% ---------------- Identify bins to exclude/include ----------------
% Fundamental bin (0-based) is k (because fin = k*fs/M)
k1 = k;  % 0-based

% Helper: build exclusion mask for bins 0..M/2
exclude = false(size(P1_ss));

% Exclude DC
exclude(1) = true; % bin 0

% Exclude fundamental guard region
exclude = mark_bins(exclude, k1, guard, Knyq);

% Exclude harmonics (for SNR only)
harm_bins = zeros(max_harm,1);
for h = 2:max_harm
    kh = mod(h*k, M);          % 0..M-1
    if kh > Knyq
        kh = M - kh;           % fold to 0..M/2 (aliasing into Nyquist band)
    end
    harm_bins(h) = kh;
    exclude = mark_bins(exclude, kh, guard, Knyq);
end

%% ---------------- Compute signal / noise / distortion powers ----------------
% Signal power = sum bins around fundamental (single-sided power)
sig_mask = false(size(P1_ss));
sig_mask = mark_bins(sig_mask, k1, guard, Knyq);
Psig = sum(P1_ss(sig_mask));

% ---- SINAD: exclude only DC + signal (all other bins are N + D) ----
sinad_excl = false(size(P1_ss));
sinad_excl(1) = true;                 % DC
sinad_excl = mark_bins(sinad_excl, k1, guard, Knyq); % fundamental
Pnd = sum(P1_ss(~sinad_excl));         % noise + distortion power
SINAD = 10*log10(Psig / (Pnd + eps));

% ---- SNR: exclude DC + signal + harmonics (remaining bins = noise only) ----
Pnoise = sum(P1_ss(~exclude));
SNR = 10*log10(Psig / (Pnoise + eps));

% ---- SFDR: worst spur excluding DC + fundamental (spur can be anywhere) ----
sfdr_excl = false(size(P1_ss));
sfdr_excl(1) = true;
sfdr_excl = mark_bins(sfdr_excl, k1, guard, Knyq);

spur_powers = P1_ss(~sfdr_excl);
Pspur = max(spur_powers);
SFDR = 10*log10(Psig / (Pspur + eps));

% ---- ENOB from SINAD ----
ENOB = (SINAD - 1.76)/6.02;

%% ---------------- Report ----------------
fprintf('fs   = %.3f GHz\n', fs/1e9);
fprintf('M    = %d\n', M);
fprintf('N    = %d bits\n', N);
fprintf('fin  = %.6f MHz (bin k=%d)\n\n', fin/1e6, k);

fprintf('SNR   = %.2f dB\n', SNR);
fprintf('SINAD = %.2f dB\n', SINAD);
fprintf('SFDR  = %.2f dB\n', SFDR);
fprintf('ENOB  = %.2f bits\n', ENOB);

%% ---------------- Plots ----------------
% Time-domain: show first few samples + quantized levels
Ns = 200;
figure; 
plot(t(1:Ns)*1e9, x(1:Ns), 'LineWidth', 1); hold on;
stairs(t(1:Ns)*1e9, y(1:Ns), 'LineWidth', 1);
grid on;
xlabel('Time (ns)'); ylabel('Voltage (V)');
title('Input vs Quantized Output (first samples)');
legend('Input x[n]', 'Quantized y[n]');

% FFT magnitude in dBFS
figure;
plot(f/1e9, mag_dbfs, 'LineWidth', 1);
grid on;
xlabel('Frequency (GHz)'); ylabel('Power (dBFS)');
title('8192-pt FFT (single-sided, dBFS)');
xlim([0 fs/(2*1e9)]);

%% ---------------- Local function ----------------
function mask = mark_bins(mask, k0, guard, Knyq)
% mark bins [k0-guard, k0+guard] in a 0..Knyq single-sided spectrum mask
    lo = max(0, k0-guard);
    hi = min(Knyq, k0+guard);
    % MATLAB indexing: bin 0 -> index 1
    mask((lo+1):(hi+1)) = true;
end
