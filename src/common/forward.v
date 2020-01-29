//-----------------------------------------------------------------------------
// <forward-fixed>
//  - Forward-propagation module
//  - Latency: (2 + log2(IN_PIXS))
//-----------------------------------------------------------------------------
// Version 4.00
//-----------------------------------------------------------------------------
// Taito Manabe, Jan 24, 2018
//-----------------------------------------------------------------------------
`default_nettype none
`timescale 1ns/1ns
  
module forward
  #( parameter integer IN_PIXS   = -1, // FLT_SIZE * FLT_SIZE * PREV_UNITS
     parameter integer OUT_PIXS  = -1, // UNITS
     parameter integer INT_BITW  = -1,
     parameter integer FRAC_BITW = -1,
     parameter [0:(INT_BITW+FRAC_BITW)*IN_PIXS*OUT_PIXS-1] FLT  = 0,
     parameter [0:(INT_BITW+FRAC_BITW*2)*OUT_PIXS-1]       BIAS = 0 )
   ( clock, in_pixels, out_pixels );

   // local parameters --------------------------------------------------------
   localparam integer IN_BITW   = INT_BITW + FRAC_BITW;
   localparam integer OUT_BITW  = INT_BITW + FRAC_BITW * 2;
   localparam integer ADD_DEPTH = log2(IN_PIXS);
   
   // inputs/outputs ----------------------------------------------------------
   input wire 	                      clock;
   input wire [0:IN_BITW*IN_PIXS-1]   in_pixels;
   output reg [0:OUT_BITW*OUT_PIXS-1] out_pixels;

   // variables ---------------------------------------------------------------
   genvar 			      ip, op, b;

   generate
      // for each output pixel
      for(op = 0; op < OUT_PIXS; op = op + 1) begin: fwd_op
	 localparam integer BASE = IN_BITW * IN_PIXS * op;
	 reg [0:OUT_BITW*IN_PIXS-1] prods;
	 // for each input pixel
	 for(ip = 0; ip < IN_PIXS; ip = ip + 1) begin: fwd_ip
	    wire signed [IN_BITW-1:0]  pixel;
	    wire signed [OUT_BITW-1:0] ext_pixel;
	    // bit extension
	    assign pixel = in_pixels[IN_BITW * ip +: IN_BITW];
	    for(b = 0; b < FRAC_BITW; b = b + 1) begin: fwd_ext_b
	       assign ext_pixel[IN_BITW+b] = pixel[IN_BITW-1];
	    end
	    assign ext_pixel[IN_BITW-1:0] = pixel;
	    // multiplication with filter coefficients
	    always @(posedge clock)
	      prods[OUT_BITW * ip +: OUT_BITW] 
		<= ext_pixel * $signed(FLT[BASE + IN_BITW * ip +: IN_BITW]);
	 end
	 // integration
	 wire signed [OUT_BITW-1:0] sum;
	 integrate
	   #( .IN_NUM(IN_PIXS), .BIT_WIDTH(OUT_BITW) )
	 itg_0
	   (  .clock(clock), .in_values(prods), .out_value(sum) );
	 // adds bias
	 always @(posedge clock)
	   out_pixels[OUT_BITW * op +: OUT_BITW] 
	     <= sum + $signed(BIAS[OUT_BITW * op +: OUT_BITW]);
      end
   endgenerate

   // common functions --------------------------------------------------------
   // calculates ceil(log2(value))
   function integer log2;
      input [63:0] value;
      begin
     	 value = value - 1;
	 for ( log2 = 0; value > 0; log2 = log2 + 1 )
	   value = value >> 1;
      end
   endfunction  

endmodule
