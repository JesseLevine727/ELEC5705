module sign_extract (
    input  wire       valid,
    input  wire       cal_mode_comp,
    input  wire       ab_sel,
    input  wire       d15,
    input  wire       d16,
    output reg        sign_valid,
    output reg        sign_pos,
    output reg        sign_neg,
    output reg        inc_i,
    output reg        dec_i
);
    always @* begin
        sign_valid = 1'b0;
        sign_pos = 1'b0;
        sign_neg = 1'b0;
        inc_i = 1'b0;
        dec_i = 1'b0;

        if (valid && (d15 ^ d16)) begin
            sign_valid = 1'b1;

            // Project sign convention:
            // sign_pos/sign_neg indicate the correction direction required by
            // the analog trim hardware, not an abstract mathematical sign.
            // For the current calibration orientation, 10 requests increment
            // and 01 requests decrement.
            sign_pos = d15 & (~d16);
            sign_neg = (~d15) & d16;

            inc_i = sign_pos;
            dec_i = sign_neg;
        end
    end
endmodule
