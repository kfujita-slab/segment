//-----------------------------------------------------------------------------
// <integrate-fixed>
//  - Calculates the summation of all input values
//  - Overall latency is (log2(IN_NUM)) clock cycles
//-----------------------------------------------------------------------------
// Version 4.00
//-----------------------------------------------------------------------------
// Taito Manabe, Jan 24, 2018
//-----------------------------------------------------------------------------
`default_nettype none
`timescale 1ns/1ns
  
module integrate
  #( parameter integer IN_NUM    = -1,
     parameter integer BIT_WIDTH = -1 )
   ( clock, in_values, out_value );

   // local parameters --------------------------------------------------------
   localparam integer ADD_DEPTH  = log2(IN_NUM);
   
   // inputs / outputs --------------------------------------------------------
   input wire         	             clock;
   input wire [0:IN_NUM*BIT_WIDTH-1] in_values;
   output wire [BIT_WIDTH-1:0] 	     out_value;
   
   // -------------------------------------------------------------------------
   integer 			     i;
   genvar 			     p, d;
   generate

      // calculates the summation of input values
      for(d = 0; d < ADD_DEPTH; d = d + 1) begin : itg_add_d
 	 for(p = 0; p < IN_NUM; p = p + (1 << (d+1))) begin : add_p   

	    localparam integer STEP = 1 << d;
	    reg [BIT_WIDTH-1:0] sum;

	    if(p + STEP < IN_NUM) begin : with_add
	       wire [BIT_WIDTH-1:0] val_1, val_2;
	       if(d == 0) begin
		  assign val_1 = in_values[   p *BIT_WIDTH +: BIT_WIDTH];
		  assign val_2 = in_values[(p+1)*BIT_WIDTH +: BIT_WIDTH];
	       end
	       else begin
		  assign val_1 = itg_add_d[d-1].add_p[p].sum;
		  assign val_2 = itg_add_d[d-1].add_p[p + STEP].sum;
	       end
	       always @(posedge clock)
		 sum <= val_1 + val_2;
	    end
	    // 1 clock cycle delay
	    else begin : no_add
	       wire [BIT_WIDTH-1:0] val;
	       if( d == 0 )
		 assign val = in_values[p*BIT_WIDTH +: BIT_WIDTH];
	       else
		 assign val = itg_add_d[d-1].add_p[p].sum;
	       always @(posedge clock)
		 sum <= val;
	    end
	 end
      end

      // assigns result --------------------------------------------------
      wire [BIT_WIDTH-1:0] result;
      if(ADD_DEPTH == 0) // if IN_NUM == 1
	assign out_value = in_values;
      else
	assign out_value = itg_add_d[ADD_DEPTH-1].add_p[0].sum;
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
