function y = quantize_nl(x, a3, Vpp, LSB, N)
    % Cubic-only front-end nonlinearity (a2=0): x_nl = x + a3*x^3
    x_nl = x + a3*(x.^3);

    % Keep within ADC input range so we measure "nonlinearity" not hard clipping
    x_nl = min(max(x_nl, -Vpp/2), (Vpp/2 - LSB));

    % Ideal uniform quantizer
    code = floor((x_nl + Vpp/2) / LSB);
    code = max(0, min(code, 2^N - 1));

    % Mid-tread reconstruction at code centers
    y = (code + 0.5)*LSB - Vpp/2;
end