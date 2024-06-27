module LMS_Filter(
    input clk,
    input reset,
    input signed [15:0] input_signal,
    input signed [15:0] desired_signal,
    input signed [15:0] mu, // Step size
    output signed [15:0] output_signal,
    output signed [15:0] error_signal
);
    parameter N = 101; // Number of filter taps
    reg signed [15:0] weights[N-1:0]; // Filter weights
    reg signed [15:0] x[N-1:0]; // Input signal delay line
    reg signed [31:0] y; // Output of the filter
    reg signed [31:0] e; // Error signal
    integer i;

    // Shift register for the input signal
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < N; i = i + 1) begin
                x[i] <= 0;
            end
        end else begin
            x[0] <= input_signal;
            for (i = 1; i < N; i = i + 1) begin
                x[i] <= x[i-1];
            end
        end
    end

    // Calculate the filter output
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            y <= 0;
        end else begin
            y <= 0;
            for (i = 0; i < N; i = i + 1) begin
                y <= y + x[i] * weights[i];
            end
        end
    end

    // Calculate the error signal
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            e <= 0;
        end else begin
            e <= desired_signal - y[31:16]; // Scale down to match input precision
        end
    end

    // Update the weights
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < N; i = i + 1) begin
                weights[i] <= 0;
            end
        end else begin
            for (i = 0; i < N; i = i + 1) begin
                weights[i] <= weights[i] + (mu * e[31:16] * x[i] >>> 15);
            end
        end
    end

    assign output_signal = y[31:16]; // Scale down to match input precision
    assign error_signal = e[31:16]; // Scale down to match input precision
endmodule
