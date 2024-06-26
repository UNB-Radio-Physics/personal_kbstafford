module fir_filter #(parameter N = 101) (
    input wire clk,
    input wire reset,
    input wire signed [15:0] x_in,
    output reg signed [15:0] y_out
);
    reg signed [15:0] delay_line [0:N-1];  // Input delay line
    integer i;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset delay line and output
            for (i = 0; i < N; i = i + 1) begin
                delay_line[i] <= 16'd0;
            end
            y_out <= 16'd0;
        end else begin
            // Shift delay line
            for (i = N-1; i > 0; i = i - 1) begin
                delay_line[i] <= delay_line[i-1];
            end
            delay_line[0] <= x_in;

            // Compute filter output with binary coefficients
            y_out <= 16'd0;
            for (i = 0; i < N; i = i + 1) begin
                if (i % 2 == 0) begin
                    y_out <= y_out + delay_line[i];  // Coefficient is 1
                end else begin
                    y_out <= y_out - delay_line[i];  // Coefficient is -1
                end
            end
        end
    end
endmodule

module lms_adaptive_filter #(parameter N = 101, parameter MU = 16'h0001) (
    input wire clk,
    input wire reset,
    input wire signed [15:0] x_in,
    input wire signed [15:0] d_in,
    output reg signed [15:0] y_out
);
    reg signed [15:0] delay_line [0:N-1];  // Input delay line
    reg signed [15:0] error;
    reg signed [15:0] coeffs [0:N-1];  // Binary coefficients for adaptive filtering
    integer i;

    initial begin
        // Initialize filter coefficients to binary values (1 or -1)
        for (i = 0; i < N; i = i + 1) begin
            if (i % 2 == 0)
                coeffs[i] = 16'd1;
            else
                coeffs[i] = -16'sd1;
        end
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset delay line and output
            for (i = 0; i < N; i = i + 1) begin
                delay_line[i] <= 16'd0;
            end
            y_out <= 16'd0;
            error <= 16'd0;
        end else begin
            // Shift delay line
            for (i = N-1; i > 0; i = i - 1) begin
                delay_line[i] <= delay_line[i-1];
            end
            delay_line[0] <= x_in;

            // Compute filter output
            y_out <= 16'd0;
            for (i = 0; i < N; i = i + 1) begin
                y_out <= y_out + (coeffs[i] * delay_line[i]);
            end

            // Compute error
            error <= d_in - y_out;

            // Update filter coefficients with binary values
            for (i = 0; i < N; i = i + 1) begin
                if (error * delay_line[i] > 0) begin
                    coeffs[i] <= 16'd1;
                end else begin
                     coeffs[i] = -16'sd1;
                end
            end
        end
    end
endmodule

module top_module (
    input wire clk,
    input wire reset,
    input wire signed [15:0] x_in,
    input wire signed [15:0] d_in,
    output wire signed [15:0] y_out
);
    wire signed [15:0] fir_out;

    // Instantiate FIR filter
    fir_filter fir (
        .clk(clk),
        .reset(reset),
        .x_in(x_in),
        .y_out(fir_out)
    );

    // Instantiate LMS adaptive filter
    lms_adaptive_filter lms (
        .clk(clk),
        .reset(reset),
        .x_in(x_in),
        .d_in(d_in),
        .y_out(y_out)
    );
endmodule


