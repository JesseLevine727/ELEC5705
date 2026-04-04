module cal_register #(
    parameter integer WIDTH = 7,
    parameter integer INIT  = 0
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             inc_o,
    input  wire             dec_o,
    output reg [WIDTH-1:0]  trim_word
);
    localparam [WIDTH-1:0] TRIM_MAX = {WIDTH{1'b1}};
    localparam [WIDTH-1:0] TRIM_MIN = {WIDTH{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            trim_word <= INIT[WIDTH-1:0];
        end else if (inc_o && !dec_o) begin
            if (trim_word != TRIM_MAX)
                trim_word <= trim_word + {{(WIDTH-1){1'b0}}, 1'b1};
        end else if (dec_o && !inc_o) begin
            if (trim_word != TRIM_MIN)
                trim_word <= trim_word - {{(WIDTH-1){1'b0}}, 1'b1};
        end
    end
endmodule
