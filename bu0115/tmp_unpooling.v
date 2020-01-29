//-----------------------------------------------------------------------------
// <layer-fixed>
//  - Composes one layer of CNN
//  - Generates network input images with the stream architecture,
//    applies flips corresponding to the coordinates, obtains results
//    using <forward> module and converts them to binary format
//-----------------------------------------------------------------------------
// Version 4.10 (June 27, 2018)
//  - Compatible with color images (YCbCr)
//  - Code refinement
//-----------------------------------------------------------------------------
// (C) 2018 Taito Manabe, all rights reserved.
//-----------------------------------------------------------------------------
// unpooling kfujita
//
`default_nettype none
`timescale 1ns/1ns

module unpooling
#(  parameter integer WIDTH         = -1,
    parameter integer HEIGHT        = -1,
    parameter integer W_WIDTH       = -1,
    parameter integer W_HEIGHT      = -1,
    parameter integer FIXED_BITW    = -1,
    parameter integer UNITS         = -1,
    parameter integer LEVEL         = -1
)
(   clock,      n_rst,
    in_pixels, in_vcnt,  in_hcnt,
    out_pixels, out_vcnt, out_hcnt );

// local parameters --------------------------------------------------------
localparam integer FLT_SIZE   = 2                   // 2x2 max pooling
localparam integer IN_PIXS    = FIT_SIZE * FIT_SIZE * UNITS;
//localparam integer LATENCY    = 3 + log2(IN_PIXS);  // meccazyuuouyade tyanntosina
localparam integer LATENCY    = 4;  /////////////////// mecca zisin naiyo!!!!
localparam integer PACTH_SIZE = FLT_SIZE;
localparam integer PATCH_BITW = PATCH_SIZE * PATCH_SIZE * FIXED_BITW;
localparam integer V_BITW     = log2(W_HEIGHT);
localparam integer H_BITW     = log2(W_WIDTH);

// local parameters for delay
localparam integer UppR_LATENCY =  1 << LEVEL;
localparam integer LowL_LATENCY = (1 << LEVEL) * W_WIDTH;
localparam integer LowR_LATENCY = UppR_LATENCY + LowL_LATENCY;
// local parameters for unpooling
localparam integer OUT_WIDTH    =  WIDTH    * 2;
localparam integer OUT_HEIGHT   =  HEIGHT   * 2;
localparam integer OUT_W_WIDTH  =  W_WIDTH  * 2;
localparam integer OUT_W_HEIGHT =  W_HEIGHT * 2;

localparam integer OUT_V_BITW   = log2(OUT_W_HEIGHT); //unpooling x2 
localparam integer OUT_H_BITW   = log2(OUT_W_WIDTH);  //unpooling x2

// inputs/outputs ----------------------------------------------------------
input wire                             clock, n_rst;
input wire [0:FIXED_BITW*UNITS-1]      in_pixels;
input wire [V_BITW-1:0]                in_vcnt;
input wire [H_BITW-1:0]                in_hcnt;
output wire [0:FIXED_BITW*UNITS-1]     out_pixels;
output wire [OUT_V_BITW-1:0]           out_vcnt;    //[log2(V_BITW/2)-1:0] ?
output wire [OUT_H_BITW-1:0]           out_hcnt;    //[log2(H_BITW/2)-1:0] ?

// -------------------------------------------------------------------------
genvar      p, v, h, m;

// buffering and generating input images -----------------------------------
generate

// patch extraction --------------------------------------------------
reg [H_BITW-1:0]             stp_hcnt;
reg [V_BITW-1:0]             stp_vcnt;
//reg [0:FIXED_BITW*IN_PIXS-1] flip_pixels;
reg [0:FIXED_BITW*UNITS-1]   pooling_pixels;

for(p = 0; p < UNITS; p = p + 1) begin : ly_patch

    // bit extension -----------------------------------------------
/*
    wire [PREV_BITW-1:0]  prev_pixel;
    wire [FIXED_BITW-1:0] new_pixel;
    assign prev_pixel = in_pixels[PREV_BITW * p +: PREV_BITW];
    if(PREV_INT_BITW < INT_BITW) begin
        localparam integer EXP_WIDTH = INT_BITW - PREV_INT_BITW;
        for(m = 0; m < EXP_WIDTH; m = m + 1) begin : ly_bitext
            assign new_pixel[PREV_BITW+m] = prev_pixel[PREV_BITW-1];
        end
        assign new_pixel[PREV_BITW-1:0] = prev_pixel;
    end
    else
        assign new_pixel = prev_pixel;
*/
    //
    wire [FIXED_BITW-1:0] new_pixel;
    assign new_pixel = in_pixels[FIXED_BITW * p +: FIXED_BITW];

    // stream patch ------------------------------------------------
    wire [0:PATCH_BITW-1] stp_patch;
    wire [H_BITW-1:0]     stp_hcnt_w;
    wire [V_BITW-1:0]     stp_vcnt_w;
    stream_patch
    #(  .BIT_WIDTH(FIXED_BITW),
        .IMAGE_HEIGHT(HEIGHT),     .IMAGE_WIDTH(WIDTH),
        .FRAME_HEIGHT(W_HEIGHT),   .FRAME_WIDTH(W_WIDTH),
        .PATCH_HEIGHT(PATCH_SIZE), .PATCH_WIDTH(PATCH_SIZE),
        .CENTER_V(PATCH_SIZE-1),   .CENTER_H(PATCH_SIZE-1), // 2x2 FLT CENTER ???
        .PADDING(0) )                                       // no padding
    stp_0
    (   .clock(clock),         .n_rst(n_rst),
        .in_pixel(new_pixel),
        .in_hcnt(in_hcnt),     .in_vcnt(in_vcnt),
        .out_patch(stp_patch),
        .out_hcnt(stp_hcnt_w), .out_vcnt(stp_vcnt_w)  );

    if(p == 0) begin
        always @(posedge clock)
            {stp_hcnt, stp_vcnt} <= {stp_hcnt_w, stp_vcnt_w}; // pooling
    end

    //////////////////////////////////////////////////////////////
    // pooling
    // LATENCY 4 ?
    //////////////////////////////////////////////////////////////
    reg signed [FIXED_BITW-1:0]  a,b,c,d;
    reg signed [FIXED_BITW-1:0]  more0,more1;
    reg [0:FIXED_BITW-1]         max;

    always @(posedge clock) begin

        a <= stp_patch[0           +:FIXED_BITW  ];
        b <= stp_patch[FIXED_BITW  +:FIXED_BITW*2];
        c <= stp_patch[FIXED_BITW*2+:FIXED_BITW*3];
        d <= stp_patch[FIXED_BITW*3+:FIXED_BITW*4];

        if(a > b)
            more0 <= a;
        else
            more0 <= b;
        if(c > d)
            more1 <= c;
        else
            more1 <= d;
        if(more0 > more1)
            max <= more0;
        else
            max <= more1;

        pooling_pixels[p * FIXED_BITW +: FIXED_BITW] <= max;
    end
    //////////////////////////////////////////////////////////////
end
endgenerate

// coordinates adjustment --------------------------------------------------
wire [H_BITW-1:0]             prev_vcnt;
wire [V_BITW-1:0]             prev_hcnt;
wire [0:FIXED_BITW*UNITS-1]   prev_out;
assign prev_out = pooling_pixels;

coord_adjuster
#(  .HEIGHT(W_HEIGHT), .WIDTH(W_WIDTH), .LATENCY(LATENCY) )
ca_1
(   .clock(clock), .in_vcnt(stp_vcnt), .in_hcnt(stp_hcnt),
    .out_vcnt(prev_vcnt), .out_hcnt(prev_hcnt) );

// check enable data & output
wire enable;
assign enable = (prev_vcnt[0]==1'b1 && prev_hcnt[0]==1'b1);

reg [H_BITW-1:0]               reg_vcnt;
reg [V_BITW-1:0]               reg_hcnt;
reg [0:FIXED_BITW*UNITS-1]     reg_out;
always @(posedge clock)begin
    if(enable) begin
        reg_vcnt <= prev_vcnt;
        reg_hcnt <= prev_hcnt;
        reg_out  <= prev_out;
    end
    else begin
        reg_vcnt <= reg_vcnt;
        reg_hcnt <= reg_hcnt;
        reg_out  <= reg_out;
    end
end

assign {out_vcnt, out_hcnt} = {reg_vcnt>>1, reg_hcnt>>1};
assign out_pixels = reg_out;

// delay for cb/cr channels ------------------------------------------------

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
