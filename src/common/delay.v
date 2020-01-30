//-----------------------------------------------------------------------------
// <delay> 
//  - Module for delaying an input (like shift registers)
//    - Latency: <LATENCY> clock cycles
//    - <LATENCY> MUST be >= 1
//-----------------------------------------------------------------------------
// Version 1.00 (Apr 27, 2018)
//  - Initial version
//-----------------------------------------------------------------------------
// (C) 2018 Taito Manabe, all rights reserved.
//-----------------------------------------------------------------------------
`default_nettype none
`timescale 1ns/1ns

module delay
#( parameter integer BIT_WIDTH = -1,
parameter integer LATENCY   = 1 )
( clock,   n_rst,   enable,
in_data, out_data );

// inputs/outputs ----------------------------------------------------------
input wire       	       clock, n_rst, enable;
input wire [BIT_WIDTH-1:0]  in_data;
output wire [BIT_WIDTH-1:0] out_data;

// fifo --------------------------------------------------------------------
generate
if(2 <= LATENCY) begin : delay_use_fifo
    fifo
    #( .BIT_WIDTH(BIT_WIDTH), .FIFO_SIZE(LATENCY), 
    .INITIAL_SIZE(LATENCY - 1) )
    fifo_0
    (  .wr_clock(clock),  .rd_clock(clock),  .n_rst(n_rst),
        .wr_en(enable),      .rd_en(1),
    .wr_data(in_data), .rd_data(out_data) );
end
else begin : delay_no_use_fifo
    reg [BIT_WIDTH-1:0] data;
    always @(posedge clock) begin
        data <= in_data;
    end
    assign out_data = data;
end
endgenerate

// functions ---------------------------------------------------------------
function integer log2;
    input integer value;
    begin
        value = value - 1;
        for ( log2 = 0; value > 0; log2 = log2 + 1 )
            value = value >> 1;
    end
endfunction

endmodule
`default_nettype wire

