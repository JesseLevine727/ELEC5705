module calibration_behavioral #(
    parameter integer COMP_BASE_ERR = 12,
    parameter integer DAC_BASE_ERR  = 10
) (
    input  wire [6:0] comp_trim,
    input  wire [6:0] dac_trim,
    input  wire       cal_mode_comp,
    input  wire       ab_sel,
    output reg        valid,
    output reg        d15,
    output reg        d16,
    output reg signed [15:0] effective_error
);
    integer signed err_now;

    always @* begin
        valid = 1'b1;

        if (cal_mode_comp) begin
            err_now = COMP_BASE_ERR - $signed({1'b0, comp_trim});
        end else begin
            err_now = DAC_BASE_ERR - $signed({1'b0, dac_trim});
        end

        effective_error = err_now;

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
        end else begin
            d15 = 1'b0;
            d16 = 1'b0;
        end
    end
endmodule
