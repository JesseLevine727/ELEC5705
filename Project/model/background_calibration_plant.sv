module background_calibration_plant #(
    parameter integer COMP_BASE_ERR = 20,
    parameter integer DAC0_BASE_ERR = 20,
    parameter integer DAC1_BASE_ERR = 18,
    parameter integer DAC2_BASE_ERR = 16,
    parameter integer DAC3_BASE_ERR = 14,
    parameter integer DAC4_BASE_ERR = 12
) (
    input  wire        valid,
    input  wire        cal_mode_comp,
    input  wire [2:0]  channel,
    input  wire        ab_sel,
    input  wire [14:0] raw_code,
    input  real        vin_norm,
    input  wire [15:0] noise_state,
    input  wire [6:0]  comp_trim,
    input  wire [6:0]  dac_trim0,
    input  wire [6:0]  dac_trim1,
    input  wire [6:0]  dac_trim2,
    input  wire [6:0]  dac_trim3,
    input  wire [6:0]  dac_trim4,
    output reg         d15,
    output reg         d16,
    output reg         noise_flip,
    output reg signed [15:0] active_error
);
    real vdac15;
    real vdac16;
    real vdac15_ideal;
    real vdac16_ideal;
    real vin_eval;
    real delta_off;
    real noise15;
    real noise16;
    real comp_scale;
    real dac_trim_scale;
    real diff_lsb;
    reg  [14:0] alt_code;
    reg  d15_ideal;
    reg  d16_ideal;

    function [14:0] with_alt_x;
        input [14:0] code_in;
        input [6:0]  x_alt;
        begin
            with_alt_x = {x_alt, code_in[7:0]};
        end
    endfunction

    function [6:0] alt_x_code;
        input [2:0] ch;
        input       dir_ab;
        begin
            case (ch)
                3'd0: alt_x_code = dir_ab ? 7'b0111111 : 7'b1000000;
                3'd1: alt_x_code = dir_ab ? 7'b0101111 : 7'b0110000;
                3'd2: alt_x_code = dir_ab ? 7'b0110111 : 7'b0111000;
                3'd3: alt_x_code = dir_ab ? 7'b0111011 : 7'b0111100;
                3'd4: alt_x_code = dir_ab ? 7'b0111101 : 7'b0111110;
                default: alt_x_code = 7'b0000000;
            endcase
        end
    endfunction

    function real bit_weight;
        input integer idx;
        input [6:0] t0;
        input [6:0] t1;
        input [6:0] t2;
        input [6:0] t3;
        input [6:0] t4;
        real ideal;
        real mismatch;
        real trim_scale;
        begin
            trim_scale = 1.0 / (8192.0 * 16.0);
            mismatch = 0.0;
            case (idx)
                14: begin ideal = 1.0/2.0;  mismatch = -(DAC0_BASE_ERR - t0) * trim_scale; end
                13: begin ideal = 1.0/4.0;  mismatch = -(DAC1_BASE_ERR - t1) * trim_scale; end
                12: begin ideal = 1.0/8.0;  mismatch = -(DAC2_BASE_ERR - t2) * trim_scale; end
                11: begin ideal = 1.0/16.0; mismatch = -(DAC3_BASE_ERR - t3) * trim_scale; end
                10: begin ideal = 1.0/32.0; mismatch = -(DAC4_BASE_ERR - t4) * trim_scale; end
                 9: ideal = 1.0/64.0;
                 8: ideal = 1.0/128.0;
                 7: ideal = 1.0/256.0;
                 6: ideal = 1.0/512.0;
                 5: ideal = 1.0/1024.0;
                 4: ideal = 1.0/2048.0;
                 3: ideal = 1.0/4096.0;
                 2: ideal = 1.0/4096.0;
                 1: ideal = 1.0/8192.0;
                 0: ideal = 1.0/8192.0;
                default: ideal = 0.0;
            endcase
            bit_weight = ideal + mismatch;
        end
    endfunction

    function real code_to_vdac;
        input [14:0] code_in;
        input [6:0] t0;
        input [6:0] t1;
        input [6:0] t2;
        input [6:0] t3;
        input [6:0] t4;
        integer idx;
        real accum;
        begin
            accum = 0.0;
            for (idx = 14; idx >= 0; idx = idx - 1) begin
                if (code_in[idx])
                    accum = accum + bit_weight(idx, t0, t1, t2, t3, t4);
            end
            code_to_vdac = accum;
        end
    endfunction

    function real code_to_vdac_ideal;
        input [14:0] code_in;
        integer idx;
        real accum;
        begin
            accum = 0.0;
            for (idx = 14; idx >= 0; idx = idx - 1) begin
                if (code_in[idx])
                    accum = accum + bit_weight(idx, 7'd0, 7'd0, 7'd0, 7'd0, 7'd0);
            end
            code_to_vdac_ideal = accum;
        end
    endfunction

    always @* begin
        comp_scale = 1.0 / (8192.0 * 4.0);
        alt_code = with_alt_x(raw_code, alt_x_code(channel, ab_sel));
        vdac15 = code_to_vdac(raw_code, dac_trim0, dac_trim1, dac_trim2, dac_trim3, dac_trim4);
        vdac16 = code_to_vdac(alt_code, dac_trim0, dac_trim1, dac_trim2, dac_trim3, dac_trim4);
        vdac15_ideal = code_to_vdac_ideal(raw_code);
        vdac16_ideal = code_to_vdac_ideal(alt_code);
        delta_off = -(COMP_BASE_ERR - comp_trim) * comp_scale;
        noise15 = ((noise_state[5:0] / 63.0) - 0.5) * (cal_mode_comp ? (2.0 / 8192.0) : (0.5 / 8192.0));
        noise16 = ((noise_state[11:6] / 63.0) - 0.5) * (cal_mode_comp ? (1.0 / 8192.0) : (0.5 / 8192.0));
        vin_eval = vin_norm;
        d15_ideal = 1'b0;
        d16_ideal = 1'b0;

        d15 = 1'b0;
        d16 = 1'b0;
        active_error = 0;
        noise_flip = 1'b0;

        if (valid) begin
            if (cal_mode_comp) begin
                // Repeat the same DAC code. Cycle 15 uses mode2, cycle 16 uses mode1.
                vin_eval = vdac15_ideal + ((noise_state[15:12] / 15.0) - 0.5) * (2.0 / 8192.0);
                d15_ideal = (vin_eval >= vdac15_ideal);
                d16_ideal = (vin_eval >= (vdac15_ideal + delta_off));
                d15 = (vin_eval + noise15 >= vdac15);
                d16 = (vin_eval + noise16 >= (vdac15 + delta_off));
                diff_lsb = delta_off * 8192.0;
            end else begin
                // Force the alternative redundant code A/B on cycle 16.
                vin_eval = 0.5 * (vdac15_ideal + vdac16_ideal) +
                          ((noise_state[15:12] / 15.0) - 0.5) * (2.0 / 8192.0);
                d15_ideal = (vin_eval >= vdac15_ideal);
                d16_ideal = (vin_eval >= vdac16_ideal);
                d15 = (vin_eval + noise15 >= vdac15);
                d16 = (vin_eval + noise16 >= vdac16);
                diff_lsb = (vdac16 - vdac15) * 8192.0;
            end
            active_error = $rtoi(diff_lsb);
            noise_flip = ((d15 != d15_ideal) || (d16 != d16_ideal));
        end
    end
endmodule
