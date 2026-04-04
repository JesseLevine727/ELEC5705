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
    integer signed err_now;

    always @* begin
        err_now = 0;
        if (cal_mode_comp) begin
            err_now = COMP_BASE_ERR - $signed({1'b0, comp_trim});
        end else begin
            case (channel)
                3'd0: err_now = DAC0_BASE_ERR - $signed({1'b0, dac_trim0});
                3'd1: err_now = DAC1_BASE_ERR - $signed({1'b0, dac_trim1});
                3'd2: err_now = DAC2_BASE_ERR - $signed({1'b0, dac_trim2});
                3'd3: err_now = DAC3_BASE_ERR - $signed({1'b0, dac_trim3});
                3'd4: err_now = DAC4_BASE_ERR - $signed({1'b0, dac_trim4});
                default: err_now = 0;
            endcase
        end

        active_error = err_now;
        noise_flip = valid && (noise_state[4:0] == 5'b00000);
        d15 = 1'b0;
        d16 = 1'b0;

        if (valid) begin
            if (err_now > 0) begin
                if (cal_mode_comp) begin
                    d15 = 1'b0;
                    d16 = 1'b1;
                end else if (!ab_sel) begin
                    d15 = 1'b0;
                    d16 = 1'b1;
                end else begin
                    d15 = 1'b1;
                    d16 = 1'b0;
                end
            end else if (err_now < 0) begin
                if (cal_mode_comp) begin
                    d15 = 1'b1;
                    d16 = 1'b0;
                end else if (!ab_sel) begin
                    d15 = 1'b1;
                    d16 = 1'b0;
                end else begin
                    d15 = 1'b0;
                    d16 = 1'b1;
                end
            end

            // Occasional wrong-sign observation to force the LPF to matter.
            if (noise_flip && (d15 ^ d16)) begin
                d15 = ~d15;
                d16 = ~d16;
            end
        end
    end
endmodule
