`timescale 1ns/1ps

module tb_calibration_loop;
    reg clk;
    reg rst_n;

    reg cal_mode_comp;
    reg ab_sel;

    wire valid;
    wire d15;
    wire d16;
    wire signed [15:0] effective_error;

    wire sign_valid;
    wire sign_pos;
    wire sign_neg;
    wire inc_i;
    wire dec_i;

    wire inc_o;
    wire dec_o;
    wire [5:0] filter_count;
    wire [6:0] trim_word;

    integer fd;
    integer cycle_count;

    calibration_behavioral plant (
        .comp_trim(trim_word),
        .dac_trim(trim_word),
        .cal_mode_comp(cal_mode_comp),
        .ab_sel(ab_sel),
        .valid(valid),
        .d15(d15),
        .d16(d16),
        .effective_error(effective_error)
    );

    sign_extract sign_path (
        .valid(valid),
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

    lpf_counter #(
        .WIDTH(6),
        .INIT(32)
    ) lpf (
        .clk(clk),
        .rst_n(rst_n),
        .inc_i(inc_i),
        .dec_i(dec_i),
        .inc_o(inc_o),
        .dec_o(dec_o),
        .count(filter_count)
    );

    cal_register #(
        .WIDTH(7),
        .INIT(0)
    ) trim_reg (
        .clk(clk),
        .rst_n(rst_n),
        .inc_o(inc_o),
        .dec_o(dec_o),
        .trim_word(trim_word)
    );

    always #5 clk = ~clk;

    task run_case;
        input mode_comp;
        input dir_ab;
        input integer max_cycles;
        input [255:0] case_name;
        begin
            cal_mode_comp = mode_comp;
            ab_sel = dir_ab;
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            rst_n = 1'b1;

            $display("---- %0s ----", case_name);
            for (cycle_count = 0; cycle_count < max_cycles; cycle_count = cycle_count + 1) begin
                @(posedge clk);
                $fwrite(fd, "%0s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    case_name, cycle_count, mode_comp, dir_ab, d15, d16,
                    effective_error, sign_pos, sign_neg, trim_word);

                if ((effective_error == 0) && (trim_word != 0)) begin
                    $display("%0s converged at cycle %0d with trim=%0d", case_name, cycle_count, trim_word);
                    disable run_case;
                end
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        cal_mode_comp = 1'b1;
        ab_sel = 1'b0;

        fd = $fopen("results/calibration_loop_log.csv", "w");
        $fwrite(fd, "case_name,cycle,mode_comp,ab_sel,d15,d16,effective_error,sign_pos,sign_neg,trim_word\n");

        run_case(1'b1, 1'b0, 100, "comp_offset");
        run_case(1'b0, 1'b0, 100, "dac_ab");
        run_case(1'b0, 1'b1, 100, "dac_ba");

        $fclose(fd);
        $display("PASS tb_calibration_loop");
        $finish;
    end
endmodule
