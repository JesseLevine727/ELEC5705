module project_calibration_digital_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [14:0] raw_code,
    input  wire        d15,
    input  wire        d16,
    output wire        cal_active,
    output wire        cal_mode_comp,
    output wire [2:0]  channel,
    output wire        ab_sel,
    output wire        sign_valid,
    output wire        sign_pos,
    output wire        sign_neg,
    output wire [5:0]  lpf_count_0,
    output wire [5:0]  lpf_count_1,
    output wire [5:0]  lpf_count_2,
    output wire [5:0]  lpf_count_3,
    output wire [5:0]  lpf_count_4,
    output wire [5:0]  lpf_count_5,
    output wire [6:0]  trim_word_0,
    output wire [6:0]  trim_word_1,
    output wire [6:0]  trim_word_2,
    output wire [6:0]  trim_word_3,
    output wire [6:0]  trim_word_4,
    output wire [6:0]  trim_word_5
);
    wire inc_i;
    wire dec_i;

    wire inc_i_0 = cal_active && (channel == 3'd0) && inc_i;
    wire dec_i_0 = cal_active && (channel == 3'd0) && dec_i;
    wire inc_i_1 = cal_active && (channel == 3'd1) && inc_i;
    wire dec_i_1 = cal_active && (channel == 3'd1) && dec_i;
    wire inc_i_2 = cal_active && (channel == 3'd2) && inc_i;
    wire dec_i_2 = cal_active && (channel == 3'd2) && dec_i;
    wire inc_i_3 = cal_active && (channel == 3'd3) && inc_i;
    wire dec_i_3 = cal_active && (channel == 3'd3) && dec_i;
    wire inc_i_4 = cal_active && (channel == 3'd4) && inc_i;
    wire dec_i_4 = cal_active && (channel == 3'd4) && dec_i;
    wire inc_i_5 = cal_active && (channel == 3'd5) && inc_i;
    wire dec_i_5 = cal_active && (channel == 3'd5) && dec_i;

    wire inc_o_0;
    wire dec_o_0;
    wire inc_o_1;
    wire dec_o_1;
    wire inc_o_2;
    wire dec_o_2;
    wire inc_o_3;
    wire dec_o_3;
    wire inc_o_4;
    wire dec_o_4;
    wire inc_o_5;
    wire dec_o_5;

    sense_force u_sense_force (
        .raw_code(raw_code),
        .cal_active(cal_active),
        .cal_mode_comp(cal_mode_comp),
        .channel(channel),
        .ab_sel(ab_sel)
    );

    sign_extract u_sign_extract (
        .valid(cal_active),
        .cal_mode_comp(cal_mode_comp),
        .ab_sel(ab_sel),
        .d15(d15),
        .d16(d16),
        .sign_valid(sign_valid),
        .sign_pos(sign_pos),
        .sign_neg(sign_neg),
        .inc_i(inc_i),
        .dec_i(dec_i)
    );

    lpf_counter #(.WIDTH(6), .INIT(32)) u_lpf_0 (
        .clk(clk), .rst_n(rst_n), .inc_i(inc_i_0), .dec_i(dec_i_0),
        .inc_o(inc_o_0), .dec_o(dec_o_0), .count(lpf_count_0)
    );
    lpf_counter #(.WIDTH(6), .INIT(32)) u_lpf_1 (
        .clk(clk), .rst_n(rst_n), .inc_i(inc_i_1), .dec_i(dec_i_1),
        .inc_o(inc_o_1), .dec_o(dec_o_1), .count(lpf_count_1)
    );
    lpf_counter #(.WIDTH(6), .INIT(32)) u_lpf_2 (
        .clk(clk), .rst_n(rst_n), .inc_i(inc_i_2), .dec_i(dec_i_2),
        .inc_o(inc_o_2), .dec_o(dec_o_2), .count(lpf_count_2)
    );
    lpf_counter #(.WIDTH(6), .INIT(32)) u_lpf_3 (
        .clk(clk), .rst_n(rst_n), .inc_i(inc_i_3), .dec_i(dec_i_3),
        .inc_o(inc_o_3), .dec_o(dec_o_3), .count(lpf_count_3)
    );
    lpf_counter #(.WIDTH(6), .INIT(32)) u_lpf_4 (
        .clk(clk), .rst_n(rst_n), .inc_i(inc_i_4), .dec_i(dec_i_4),
        .inc_o(inc_o_4), .dec_o(dec_o_4), .count(lpf_count_4)
    );
    lpf_counter #(.WIDTH(6), .INIT(32)) u_lpf_5 (
        .clk(clk), .rst_n(rst_n), .inc_i(inc_i_5), .dec_i(dec_i_5),
        .inc_o(inc_o_5), .dec_o(dec_o_5), .count(lpf_count_5)
    );

    cal_register #(.WIDTH(7), .INIT(0)) u_cal_0 (
        .clk(clk), .rst_n(rst_n), .inc_o(inc_o_0), .dec_o(dec_o_0), .trim_word(trim_word_0)
    );
    cal_register #(.WIDTH(7), .INIT(0)) u_cal_1 (
        .clk(clk), .rst_n(rst_n), .inc_o(inc_o_1), .dec_o(dec_o_1), .trim_word(trim_word_1)
    );
    cal_register #(.WIDTH(7), .INIT(0)) u_cal_2 (
        .clk(clk), .rst_n(rst_n), .inc_o(inc_o_2), .dec_o(dec_o_2), .trim_word(trim_word_2)
    );
    cal_register #(.WIDTH(7), .INIT(0)) u_cal_3 (
        .clk(clk), .rst_n(rst_n), .inc_o(inc_o_3), .dec_o(dec_o_3), .trim_word(trim_word_3)
    );
    cal_register #(.WIDTH(7), .INIT(0)) u_cal_4 (
        .clk(clk), .rst_n(rst_n), .inc_o(inc_o_4), .dec_o(dec_o_4), .trim_word(trim_word_4)
    );
    cal_register #(.WIDTH(7), .INIT(0)) u_cal_5 (
        .clk(clk), .rst_n(rst_n), .inc_o(inc_o_5), .dec_o(dec_o_5), .trim_word(trim_word_5)
    );
endmodule
