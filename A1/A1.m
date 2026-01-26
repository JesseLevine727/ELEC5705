% Assignment 1 - 8-bit ADC Characterizatiom
clear;
clc;
close all;

% 1. Parameters
Fs = 10e9;              % Sampling Rate: 10 GS/s [cite: 785]
Nbits = 8;              % Resolution: 8-bit [cite: 785]
Nfft = 8192;            % FFT points [cite: 786]
Vfs = 2;                % Full Scale Voltage Range (e.g., -1V to +1V)

% 2. Generate Input Sequence (Coherent Sampling)
% Prime number of cycles (M) to ensure coherent sampling

M = 211;                % Prime number
Fin = (Fs*M / Nfft);  % Coherent Input Frequency

t = (0:Nfft-1) / Fs;    % Time vector

% Generate ideal input signal (using full scale range)
Ain = Vfs / 2;          % Amplitude (half of full scale)
vin = Ain * sin(2*pi*Fin*t);

% 3. Quantization
% Calculate LSB size
LSB = Vfs / 2^Nbits;

% Quantize the signal
% 1. Scale to LSBs
% 2. Round to nearest integer (quantization)
% 3. Saturate to min/max codes (clipping protection)
vin_codes = round(vin / LSB);
max_code = 2^(Nbits-1) - 1;
min_code = -2^(Nbits-1);
vin_codes(vin_codes > max_code) = max_code;
vin_codes(vin_codes < min_code) = min_code;

% Convert back to voltage for analysis
vout = vin_codes * LSB;

% 4. FFT
spec = fft(vout);
% One-sided PSD in dB
spec = spec(1:Nfft/2);                 % Keep positive frequencies
mag = abs(spec);                       % Magnitude
mag = mag / max(mag);                  % Normalize to Carrier (Fundamental)
mag_db = 20*log10(mag + 1e-12);        % Convert to dB (add small offset to avoid log(0))

freq = linspace(0, Fs/2, Nfft/2);      % Frequency Vector

% 5. Metric Calculation (SNR, SFDR, SINAD, ENOB)

% A. Find Signal Bin
[val, idx_fund] = max(mag); % Index of fundamental frequency

% B. Calculate Power of Signal and Noise+Distortion
% Since we normalized to the carrier, Signal Power = 1 (0 dB)
P_signal = sum(mag(idx_fund-2:idx_fund+2).^2); % Sum power around fundamental (leakage margin)
P_total = sum(mag.^2);                         % Total Power
P_noise_distortion = P_total - P_signal;       % Everything else is noise/distortion

% Calculate SINAD 
SINAD = 10*log10(P_signal / P_noise_distortion);

% C. Calculate ENOB 
ENOB = (SINAD - 1.76) / 6.02;

% D. Calculate SFDR 
% Remove fundamental region to find the next highest spur
mag_no_signal = mag;
mag_no_signal(max(1, idx_fund-5):min(length(mag), idx_fund+5)) = 0; 
[val_spur, idx_spur] = max(mag_no_signal);
SFDR = -20*log10(val_spur); % Since signal is normalized to 0dB

% E. Calculate SNR (Exclude Harmonics) 
% Identify aliased harmonic locations: | +/- K*Fs +/- n*Fa |
harmonic_indices = [];
for n = 2:10 % Check first 9 harmonics
    f_harm = mod(n * Fin, Fs);
    if f_harm > Fs/2
        f_harm = Fs - f_harm;
    end
    % Find bin index for this frequency
    [~, idx_h] = min(abs(freq - f_harm));
    harmonic_indices = [harmonic_indices, idx_h];
end

% Remove power at harmonic bins from the Noise+Distortion power
P_harmonics = 0;
for k = 1:length(harmonic_indices)
    idx = harmonic_indices(k);
    % Sum power in the harmonic bin and neighbors
    range = max(1, idx-2):min(length(mag), idx+2);
    P_harmonics = P_harmonics + sum(mag(range).^2);
end

P_noise_only = P_noise_distortion - P_harmonics;
SNR = 10*log10(P_signal / P_noise_only);

%% 6. Display Results and Plots
fprintf('---------------------------------\n');
fprintf('Analysis Results (Ideal 8-bit ADC)\n');
fprintf('---------------------------------\n');
fprintf('SINAD : %.2f dB\n', SINAD);
fprintf('ENOB  : %.2f bits\n', ENOB);
fprintf('SNR   : %.2f dB\n', SNR);
fprintf('SFDR  : %.2f dBc\n', SFDR);

figure;
plot(freq/1e6, mag_db);
grid on;
title('FFT Spectrum of Ideal 8-bit ADC');
xlabel('Frequency (MHz)');
ylabel('Magnitude (dBFS)');
ylim([-120 10]);


% 7. Plot Input vs Quantized Waveform (Time Domain)
figure;
% Plot the Ideal Analog Input
plot(t*1e9, vin, 'b', 'LineWidth', 1.5, 'DisplayName', 'Analog Input'); 
hold on;

% Plot the Quantized Output using 'stairs' to emphasize the discrete levels
stairs(t*1e9, vout, 'r--', 'LineWidth', 1, 'DisplayName', 'Quantized Output');

hold off;
grid on;
legend('show');
xlabel('Time (ns)');
ylabel('Voltage (V)');
title('Time Domain: Input vs Quantized Output (Zoomed)');

% CRITICAL: Zoom in to show only the first 3 cycles so you can see the steps
% Calculate the period of the input sine wave
T_period = 1/Fin; 
% Limit the X-axis to 3 periods (converted to nanoseconds)
xlim([0, 3*T_period*1e9]);