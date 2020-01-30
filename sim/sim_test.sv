`timescale 1ns/1ns

module sim_segment();

logic signed [23:0] in_y;
logic signed [9:0]  in_hcnt;
logic signed [8:0]  in_vcnt;

wire         [1:0]  out_y;
wire         [9:0]  out_hcnt;
wire         [8:0]  out_vcnt;

integer i;
integer j;
logic clk;
logic n_rst;
top top_inst(
    .clock(clk),
    .n_rst(n_rst),
    .in_y(in_y),
    .in_hcnt(in_hcnt),
    .in_vcnt(in_vcnt),
    .out_y(out_y),
    .out_hcnt(out_hcnt),
    .out_vcnt(out_vcnt)
);

localparam real CLOCK_FREQ = 100 * 10 ** 6; // 100 MHz
localparam real CLOCK_PERIOD_NS = 10 ** 9 / CLOCK_FREQ; // ns
localparam integer SIM_CYCLES = 270000;
initial begin
    clk <= 1'b0;
    repeat (SIM_CYCLES) begin
        #(CLOCK_PERIOD_NS / 2.0)
        clk <= 1'b0;
        #(CLOCK_PERIOD_NS / 2.0)
        clk <= 1'b1;
    end
    $finish;
end

initial begin
    n_rst <= 1'b1;
    #(CLOCK_PERIOD_NS / 2.0)
    n_rst <= 1'b0;
    #(CLOCK_PERIOD_NS / 2.0)
    n_rst <= 1'b1;
end

initial begin
    #(CLOCK_PERIOD_NS)
    #(CLOCK_PERIOD_NS)
    #(CLOCK_PERIOD_NS)
    #(CLOCK_PERIOD_NS)
    in_y <= 23'b111111110000000011111111;
    in_hcnt <= ;
    in_vcnt <= ;
    #(CLOCK_PERIOD_NS)
    print();
    
end

task print();
    $write("out_y = %d",out_y);
endtask

endmodule
