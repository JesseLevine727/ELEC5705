module lpf_counter #(
    parameter integer WIDTH = 6,
    parameter integer INIT  = 32
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             inc_i,
    input  wire             dec_i,
    output reg              inc_o,
    output reg              dec_o,
    output reg [WIDTH-1:0]  count
);
    localparam [WIDTH-1:0] COUNT_MAX = {WIDTH{1'b1}};
    localparam [WIDTH-1:0] COUNT_MIN = {WIDTH{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= INIT[WIDTH-1:0];
            inc_o <= 1'b0;
            dec_o <= 1'b0;
        end else begin
            inc_o <= 1'b0;
            dec_o <= 1'b0;

            if (inc_i && !dec_i) begin
                if (count == COUNT_MAX - {{(WIDTH-1){1'b0}}, 1'b1}) begin
                    inc_o <= 1'b1;
                    count <= INIT[WIDTH-1:0];
                end else if (count != COUNT_MAX) begin
                    count <= count + {{(WIDTH-1){1'b0}}, 1'b1};
                end
            end else if (dec_i && !inc_i) begin
                if (count == COUNT_MIN + {{(WIDTH-1){1'b0}}, 1'b1}) begin
                    dec_o <= 1'b1;
                    count <= INIT[WIDTH-1:0];
                end else if (count != COUNT_MIN) begin
                    count <= count - {{(WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    end
endmodule
