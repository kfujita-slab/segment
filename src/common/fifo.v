//-----------------------------------------------------------------------------
// <fifo> 
//  - First-in/First-out module
//  - Actual FIFO size is <FIFO_SIZE - 1>, NOT <FIFO_SIZE>
//-----------------------------------------------------------------------------
// Version 1.00
//  - Initial version
//-----------------------------------------------------------------------------
// Taito Manabe, Mar 30, 2016
//-----------------------------------------------------------------------------

`default_nettype none
`timescale 1ns/1ns
  
module fifo
  #(
    parameter integer BIT_WIDTH    = -1,
    parameter integer FIFO_SIZE    = -1,  // must be 2^n
    parameter integer INITIAL_SIZE = 0
    )
   (
    wr_clock, rd_clock, n_rst,
    wr_en,    rd_en,
    wr_data,  rd_data,
    count
    );

   // local parameters --------------------------------------------------------
   localparam integer FIFO_SIZE_P = 1 << log2(FIFO_SIZE);
   localparam integer ADDR_BITW   = log2(FIFO_SIZE_P);

   // inputs ------------------------------------------------------------------
   input wire       	       wr_clock, rd_clock, n_rst;
   input wire 		       wr_en,    rd_en;
   input wire [BIT_WIDTH-1:0]  wr_data;
   
   // outputs -----------------------------------------------------------------
   output wire [BIT_WIDTH-1:0] rd_data;
   output wire [ADDR_BITW-1:0] count;

   // registers ---------------------------------------------------------------
   reg [ADDR_BITW-1:0] 	       wr_addr,  rd_addr;
   reg 			       rd_en_buf;
   always @(posedge rd_clock)
     rd_en_buf <= rd_en;

   // ram ---------------------------------------------------------------------
   wire [BIT_WIDTH-1:0]        rd_data_tmp;
   ram
     #(
       .WORD_SIZE(BIT_WIDTH), .RAM_SIZE(FIFO_SIZE_P)
       )
   ram_0
     (
      .wr_clock(wr_clock), .rd_clock(rd_clock),  .wr_en(wr_en),
      .wr_addr(wr_addr),   .wr_data(wr_data),
      .rd_addr(rd_addr),   .rd_data(rd_data_tmp)
      );

   assign rd_data = rd_en_buf ? rd_data_tmp : 0;
   assign count   = wr_addr - rd_addr + 
		    ((wr_addr < rd_addr) ? FIFO_SIZE_P : 0);

   // -------------------------------------------------------------------------
   always @(posedge wr_clock or negedge n_rst) begin
      if(!n_rst)
	wr_addr <= INITIAL_SIZE;
      else begin
	 if(wr_en)
	   wr_addr <= wr_addr + 1;
      end
   end

   always @(posedge rd_clock or negedge n_rst) begin
      if(!n_rst)
	rd_addr <= 0;
      else begin
	 if(rd_en)
	   rd_addr <= rd_addr + 1;
      end
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
