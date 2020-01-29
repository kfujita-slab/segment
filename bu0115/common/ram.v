//-----------------------------------------------------------------------------
// <ram> 
//  - Random Access Memory module without reset
//  - Hopes that this module will be implemented using block RAMs
//  - DO NOT try to write on/read from the same address at the same time!
//-----------------------------------------------------------------------------
// Version 1.00
//  - Initial version
//-----------------------------------------------------------------------------
// Taito Manabe, Mar 30, 2016
//-----------------------------------------------------------------------------

`default_nettype none
`timescale 1ns/1ns
  
module ram
  #(
    parameter integer WORD_SIZE = -1,
    parameter integer RAM_SIZE  = -1
    )
   (
    wr_clock, rd_clock, wr_en,
    wr_addr,  wr_data,
    rd_addr,  rd_data
    );

   // local parameters --------------------------------------------------------
   localparam integer ADDR_BITW = log2(RAM_SIZE);

   // inputs ------------------------------------------------------------------
   input wire 	              wr_clock, rd_clock, wr_en;
   input wire [ADDR_BITW-1:0] wr_addr,  rd_addr;
   input wire [WORD_SIZE-1:0] wr_data;

   // outputs -----------------------------------------------------------------
   output reg [WORD_SIZE-1:0] rd_data;

   // registers ---------------------------------------------------------------
   reg [WORD_SIZE-1:0] 	      memory [0:RAM_SIZE-1];

   always @(posedge wr_clock) begin
      if(wr_en)
	memory[wr_addr] <= wr_data;
   end

   always @(posedge rd_clock) begin
      rd_data <= memory[rd_addr];
   end
   
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
