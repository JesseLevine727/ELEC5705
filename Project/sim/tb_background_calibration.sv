`timescale 1ns/1ps

module tb_background_calibration;
    reg clk;
    reg rst_n;

    wire [14:0] raw_code;
    wire [15:0] noise_state;
    wire [12:0] final_code;
    real vin_norm;

    wire cal_active;
    wire cal_mode_comp;
    wire [2:0] channel;
    wire ab_sel;

    wire d15;
    wire d16;
    wire noise_flip;
    wire signed [15:0] active_error;

    wire sign_valid;
    wire sign_pos;
    wire sign_neg;
    wire inc_i;
    wire dec_i;

    wire inc_i_0 = cal_active && (channel == 3'd0) ? inc_i : 1'b0;
    wire dec_i_0 = cal_active && (channel == 3'd0) ? dec_i : 1'b0;
    wire inc_i_1 = cal_active && (channel == 3'd1) ? inc_i : 1'b0;
    wire dec_i_1 = cal_active && (channel == 3'd1) ? dec_i : 1'b0;
    wire inc_i_2 = cal_active && (channel == 3'd2) ? inc_i : 1'b0;
    wire dec_i_2 = cal_active && (channel == 3'd2) ? dec_i : 1'b0;
    wire inc_i_3 = cal_active && (channel == 3'd3) ? inc_i : 1'b0;
    wire dec_i_3 = cal_active && (channel == 3'd3) ? dec_i : 1'b0;
    wire inc_i_4 = cal_active && (channel == 3'd4) ? inc_i : 1'b0;
    wire dec_i_4 = cal_active && (channel == 3'd4) ? dec_i : 1'b0;
    wire inc_i_5 = cal_active && (channel == 3'd5) ? inc_i : 1'b0;
    wire dec_i_5 = cal_active && (channel == 3'd5) ? dec_i : 1'b0;

    wire inc_o_0, dec_o_0, inc_o_1, dec_o_1, inc_o_2, dec_o_2;
    wire inc_o_3, dec_o_3, inc_o_4, dec_o_4, inc_o_5, dec_o_5;
    wire [5:0] count_0, count_1, count_2, count_3, count_4, count_5;
    wire [6:0] trim_0, trim_1, trim_2, trim_3, trim_4, trim_5;

    integer fd;
    integer cycle_count;
    integer active_count;
    integer comp_done_cycle;
    integer dac0_done_cycle;
    integer dac1_done_cycle;
    integer dac2_done_cycle;
    integer dac3_done_cycle;
    integer dac4_done_cycle;

    behavioral_sar_raw code_gen (
        .clk(clk),
        .rst_n(rst_n),
        .raw_code(raw_code),
        .noise_state(noise_state),
        .final_code(final_code),
        .vin_norm(vin_norm)
    );

    sense_force sf (
        .raw_code(raw_code),
        .cal_active(cal_active),
        .cal_mode_comp(cal_mode_comp),
        .channel(channel),
        .ab_sel(ab_sel)
    );

    background_calibration_plant plant (
        .valid(cal_active),
        .cal_mode_comp(cal_mode_comp),
        .channel(channel),
        .ab_sel(ab_sel),
        .noise_state(noise_state),
        .comp_trim(trim_5),
        .dac_trim0(trim_0),
        .dac_trim1(trim_1),
        .dac_trim2(trim_2),
        .dac_trim3(trim_3),
        .dac_trim4(trim_4),
        .d15(d15),
        .d16(d16),
        .noise_flip(noise_flip),
        .active_error(active_error)
    );

    sign_extract sign_path (
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

    lpf_counter #(.WIDTH(6), .INIT(32)) lpf0 (.clk(clk), .rst_n(rst_n), .inc_i(inc_i_0), .dec_i(dec_i_0), .inc_o(inc_o_0), .dec_o(dec_o_0), .count(count_0));
    lpf_counter #(.WIDTH(6), .INIT(32)) lpf1 (.clk(clk), .rst_n(rst_n), .inc_i(inc_i_1), .dec_i(dec_i_1), .inc_o(inc_o_1), .dec_o(dec_o_1), .count(count_1));
    lpf_counter #(.WIDTH(6), .INIT(32)) lpf2 (.clk(clk), .rst_n(rst_n), .inc_i(inc_i_2), .dec_i(dec_i_2), .inc_o(inc_o_2), .dec_o(dec_o_2), .count(count_2));
    lpf_counter #(.WIDTH(6), .INIT(32)) lpf3 (.clk(clk), .rst_n(rst_n), .inc_i(inc_i_3), .dec_i(dec_i_3), .inc_o(inc_o_3), .dec_o(dec_o_3), .count(count_3));
    lpf_counter #(.WIDTH(6), .INIT(32)) lpf4 (.clk(clk), .rst_n(rst_n), .inc_i(inc_i_4), .dec_i(dec_i_4), .inc_o(inc_o_4), .dec_o(dec_o_4), .count(count_4));
    lpf_counter #(.WIDTH(6), .INIT(32)) lpf5 (.clk(clk), .rst_n(rst_n), .inc_i(inc_i_5), .dec_i(dec_i_5), .inc_o(inc_o_5), .dec_o(dec_o_5), .count(count_5));

    cal_register #(.WIDTH(7), .INIT(0)) reg0 (.clk(clk), .rst_n(rst_n), .inc_o(inc_o_0), .dec_o(dec_o_0), .trim_word(trim_0));
    cal_register #(.WIDTH(7), .INIT(0)) reg1 (.clk(clk), .rst_n(rst_n), .inc_o(inc_o_1), .dec_o(dec_o_1), .trim_word(trim_1));
    cal_register #(.WIDTH(7), .INIT(0)) reg2 (.clk(clk), .rst_n(rst_n), .inc_o(inc_o_2), .dec_o(dec_o_2), .trim_word(trim_2));
    cal_register #(.WIDTH(7), .INIT(0)) reg3 (.clk(clk), .rst_n(rst_n), .inc_o(inc_o_3), .dec_o(dec_o_3), .trim_word(trim_3));
    cal_register #(.WIDTH(7), .INIT(0)) reg4 (.clk(clk), .rst_n(rst_n), .inc_o(inc_o_4), .dec_o(dec_o_4), .trim_word(trim_4));
    cal_register #(.WIDTH(7), .INIT(0)) reg5 (.clk(clk), .rst_n(rst_n), .inc_o(inc_o_5), .dec_o(dec_o_5), .trim_word(trim_5));

    always #5 clk = ~clk;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        active_count = 0;
        comp_done_cycle = -1;
        dac0_done_cycle = -1;
        dac1_done_cycle = -1;
        dac2_done_cycle = -1;
        dac3_done_cycle = -1;
        dac4_done_cycle = -1;

        fd = $fopen("results/background_calibration_log.csv", "w");
        $fwrite(fd, "cycle,raw_code,final_code,vin_norm,cal_active,mode_comp,channel,ab_sel,d15,d16,noise_flip,error,trim0,trim1,trim2,trim3,trim4,trim5,count0,count1,count2,count3,count4,count5\n");

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        begin : finish_loop
        for (cycle_count = 0; cycle_count < 600000; cycle_count = cycle_count + 1) begin
            @(posedge clk);
            if (cal_active)
                active_count = active_count + 1;

            $fwrite(fd, "%0d,%0h,%0d,%.6f,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                cycle_count, raw_code, final_code, vin_norm, cal_active, cal_mode_comp, channel, ab_sel, d15, d16, noise_flip, active_error,
                trim_0, trim_1, trim_2, trim_3, trim_4, trim_5,
                count_0, count_1, count_2, count_3, count_4, count_5);

            if ((comp_done_cycle < 0) && (trim_5 == 7'd20))
                comp_done_cycle = cycle_count;
            if ((dac0_done_cycle < 0) && (trim_0 == 7'd20))
                dac0_done_cycle = cycle_count;
            if ((dac1_done_cycle < 0) && (trim_1 == 7'd18))
                dac1_done_cycle = cycle_count;
            if ((dac2_done_cycle < 0) && (trim_2 == 7'd16))
                dac2_done_cycle = cycle_count;
            if ((dac3_done_cycle < 0) && (trim_3 == 7'd14))
                dac3_done_cycle = cycle_count;
            if ((dac4_done_cycle < 0) && (trim_4 == 7'd12))
                dac4_done_cycle = cycle_count;

            if ((comp_done_cycle >= 0) && (dac0_done_cycle >= 0) && (dac1_done_cycle >= 0) &&
                (dac2_done_cycle >= 0) && (dac3_done_cycle >= 0) && (dac4_done_cycle >= 0)) begin
                $display("All calibration loops converged by cycle %0d", cycle_count);
                disable finish_loop;
            end
        end
        end

        $display("Active calibration events: %0d", active_count);
        $display("Convergence cycles: comp=%0d dac0=%0d dac1=%0d dac2=%0d dac3=%0d dac4=%0d",
            comp_done_cycle, dac0_done_cycle, dac1_done_cycle, dac2_done_cycle, dac3_done_cycle, dac4_done_cycle);
        $fclose(fd);
        $finish;
    end
endmodule
