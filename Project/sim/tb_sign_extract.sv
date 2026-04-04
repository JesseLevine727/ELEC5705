`timescale 1ns/1ps

module tb_sign_extract;
    reg valid;
    reg cal_mode_comp;
    reg ab_sel;
    reg d15;
    reg d16;
    wire sign_valid;
    wire sign_pos;
    wire sign_neg;
    wire inc_i;
    wire dec_i;

    integer failures;

    sign_extract dut (
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

    task check_case;
        input exp_valid;
        input exp_pos;
        input exp_neg;
        begin
            #1;
            if (sign_valid !== exp_valid || sign_pos !== exp_pos || sign_neg !== exp_neg) begin
                $display("FAIL valid=%0b mode=%0b ab_sel=%0b d15=%0b d16=%0b -> got valid=%0b pos=%0b neg=%0b expected valid=%0b pos=%0b neg=%0b",
                    valid, cal_mode_comp, ab_sel, d15, d16, sign_valid, sign_pos, sign_neg, exp_valid, exp_pos, exp_neg);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;

        valid = 0; cal_mode_comp = 1; ab_sel = 0; d15 = 0; d16 = 1;
        check_case(0, 0, 0);

        valid = 1; cal_mode_comp = 1; d15 = 0; d16 = 1;
        check_case(1, 0, 1);

        valid = 1; cal_mode_comp = 1; d15 = 1; d16 = 0;
        check_case(1, 1, 0);

        valid = 1; cal_mode_comp = 1; d15 = 1; d16 = 1;
        check_case(0, 0, 0);

        valid = 1; cal_mode_comp = 0; ab_sel = 0; d15 = 0; d16 = 1;
        check_case(1, 0, 1);

        valid = 1; cal_mode_comp = 0; ab_sel = 0; d15 = 1; d16 = 0;
        check_case(1, 1, 0);

        valid = 1; cal_mode_comp = 0; ab_sel = 1; d15 = 0; d16 = 1;
        check_case(1, 0, 1);

        valid = 1; cal_mode_comp = 0; ab_sel = 1; d15 = 1; d16 = 0;
        check_case(1, 1, 0);

        if (failures == 0)
            $display("PASS tb_sign_extract");
        else
            $fatal(1, "tb_sign_extract had %0d failure(s)", failures);
    end
endmodule
