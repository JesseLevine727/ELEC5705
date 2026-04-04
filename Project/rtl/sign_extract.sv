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

            if (cal_mode_comp) begin
                // Comparator calibration:
                // 01 => positive offset difference, 10 => negative.
                sign_pos = (~d15) & d16;
                sign_neg = d15 & (~d16);
            end else begin
                // DAC calibration:
                // ab_sel=0 means cycle15 uses A and cycle16 uses B.
                // ab_sel=1 means cycle15 uses B and cycle16 uses A.
                if (!ab_sel) begin
                    sign_pos = (~d15) & d16;
                    sign_neg = d15 & (~d16);
                end else begin
                    sign_pos = d15 & (~d16);
                    sign_neg = (~d15) & d16;
                end
            end

            inc_i = sign_pos;
            dec_i = sign_neg;
        end
    end
endmodule
