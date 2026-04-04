module sense_force (
    input  wire [14:0] raw_code,
    output reg         cal_active,
    output reg         cal_mode_comp,
    output reg  [2:0]  channel,
    output reg         ab_sel
);
    wire [6:0] x_code = raw_code[14:8];
    wire [2:0] y_code = raw_code[7:5];

    always @* begin
        cal_active = 1'b0;
        cal_mode_comp = 1'b0;
        channel = 3'd0;
        ab_sel = 1'b0;

        // Paper-like activation gating:
        // Y-code must be 110. Comparator uses 11000xx.
        // DAC channels use five A/B code pairs centered near midscale.
        if (y_code == 3'b110) begin
            if (x_code[6:2] == 5'b11000) begin
                cal_active = 1'b1;
                cal_mode_comp = 1'b1;
                channel = 3'd5;
                ab_sel = x_code[1];
            end else begin
                case (x_code)
                    7'b0111111: begin cal_active = 1'b1; channel = 3'd0; ab_sel = 1'b0; end
                    7'b1000000: begin cal_active = 1'b1; channel = 3'd0; ab_sel = 1'b1; end
                    7'b0101111: begin cal_active = 1'b1; channel = 3'd1; ab_sel = 1'b0; end
                    7'b0110000: begin cal_active = 1'b1; channel = 3'd1; ab_sel = 1'b1; end
                    7'b0110111: begin cal_active = 1'b1; channel = 3'd2; ab_sel = 1'b0; end
                    7'b0111000: begin cal_active = 1'b1; channel = 3'd2; ab_sel = 1'b1; end
                    7'b0111011: begin cal_active = 1'b1; channel = 3'd3; ab_sel = 1'b0; end
                    7'b0111100: begin cal_active = 1'b1; channel = 3'd3; ab_sel = 1'b1; end
                    7'b0111101: begin cal_active = 1'b1; channel = 3'd4; ab_sel = 1'b0; end
                    7'b0111110: begin cal_active = 1'b1; channel = 3'd4; ab_sel = 1'b1; end
                    default: begin end
                endcase
            end
        end
    end
endmodule
