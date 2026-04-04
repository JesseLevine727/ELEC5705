module raw_code_stream (
    input  wire        clk,
    input  wire        rst_n,
    output reg [14:0]  raw_code,
    output reg [15:0]  noise_state
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            raw_code <= 15'h4D2B;
            noise_state <= 16'hACE1;
        end else begin
            raw_code <= {raw_code[13:0], raw_code[14] ^ raw_code[13]};
            noise_state <= {noise_state[14:0], noise_state[15] ^ noise_state[13] ^ noise_state[12] ^ noise_state[10]};
        end
    end
endmodule
