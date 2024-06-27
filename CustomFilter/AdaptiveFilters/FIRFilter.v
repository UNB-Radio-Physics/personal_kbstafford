module FIRFilter (
    input signed [15:0] sample_in,
    input signed [15:0] coefficients [0:100],  // Example: 101 taps
    output reg signed [31:0] filtered_sample
);

reg signed [31:0] accumulator;

integer i;

always @(*) begin
    accumulator = 0;  // Clear accumulator
    for (i = 0; i < 101; i++) begin
        accumulator = accumulator + coefficients[i] * sample_in[i];
    end
    filtered_sample = accumulator;
end

endmodule

