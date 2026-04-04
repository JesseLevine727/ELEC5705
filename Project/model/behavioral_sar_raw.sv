module behavioral_sar_raw (
    input  wire        clk,
    input  wire        rst_n,
    output reg [14:0]  raw_code,
    output reg [15:0]  noise_state,
    output reg [12:0]  final_code,
    output real        vin_norm
);
    integer quant_code;
    reg [12:0] code_q;
    reg        red_hi;
    reg        red_lo;
    real       vin_r;
    real vin_next;

    assign vin_norm = vin_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            raw_code <= 15'd0;
            final_code <= 13'd0;
            noise_state <= 16'hACE1;
            vin_r <= 0.5;
        end else begin
            // Broad pseudo-random analog stimulus so the centered trigger
            // codes used by the paper's background calibration occur often
            // enough without forcing calibration every cycle.
            vin_next = (noise_state + 0.5) / 65536.0;

            quant_code = $rtoi(vin_next * 8192.0);
            if (quant_code < 0)
                quant_code = 0;
            if (quant_code > 8191)
                quant_code = 8191;

            code_q = quant_code[12:0];
            red_hi = code_q[3] ^ code_q[2];
            red_lo = code_q[1] ^ code_q[0];

            // Simplified 15-bit raw-code model:
            // x-code and y-code come directly from the 13-bit conversion code,
            // while two extra redundant bits are inserted into the z-field.
            raw_code <= {
                code_q[12:6],    // 7-bit x-code
                code_q[5:3],     // 3-bit y-code
                red_hi,          // simplified redundancy bit near the 11th cycle
                code_q[2:0],
                red_lo           // simplified redundancy bit near the 7th cycle
            };
            final_code <= code_q;
            noise_state <= {noise_state[14:0], noise_state[15] ^ noise_state[13] ^ noise_state[12] ^ noise_state[10]};
            vin_r <= vin_next;
        end
    end
endmodule
