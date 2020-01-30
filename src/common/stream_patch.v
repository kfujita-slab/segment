//-----------------------------------------------------------------------------
// <stream_patch> 
//  - Extracts a patch from an input image for stream processing
//    - <PATCH_WIDTH> must be less than <WIDTH>
//-----------------------------------------------------------------------------
// Version 1.11 (Nov. 12, 2018)
//  - Added padding function
//  - Changed some parameter names
//-----------------------------------------------------------------------------
// (C) 2018 Taito Manabe. All rights reserved.
//-----------------------------------------------------------------------------
// kfujita 12_23,2019
// add enable wire for stall
`default_nettype none
`timescale 1ns/1ns

module stream_patch
#( parameter integer BIT_WIDTH    = -1, // image bit width
    parameter integer IMAGE_HEIGHT = -1, // | image size
    parameter integer IMAGE_WIDTH  = -1, // |
    parameter integer FRAME_HEIGHT = -1, //   | frame size (including sync)
    parameter integer FRAME_WIDTH  = -1, //   |
    parameter integer PATCH_HEIGHT = -1, // | patch size
    parameter integer PATCH_WIDTH  = -1, // |
    parameter integer CENTER_V     = PATCH_HEIGHT / 2, // | center position
    parameter integer CENTER_H     = PATCH_WIDTH  / 2, // |
parameter integer PADDING      = 1 ) // to apply padding or not
( clock,     n_rst,    enable,
    in_pixel,  in_vcnt,  in_hcnt,
out_patch, out_vcnt, out_hcnt );

// local parameters --------------------------------------------------------
localparam integer V_BITW     = log2(FRAME_HEIGHT);
localparam integer H_BITW     = log2(FRAME_WIDTH);
localparam integer PATCH_BITW = BIT_WIDTH * PATCH_WIDTH * PATCH_HEIGHT;

// inputs ------------------------------------------------------------------
input wire                  clock, n_rst, enable;
input wire [BIT_WIDTH-1:0]  in_pixel;
input wire [V_BITW-1:0]     in_vcnt;
input wire [H_BITW-1:0]     in_hcnt;

// outputs -----------------------------------------------------------------
output reg [0:PATCH_BITW-1] out_patch;
// example where the patch size is (3, 3) and the bit width is 8:
//
//   \ h  0 1 2 
//   v\  _______    ________a________ ____b____ ____c____ ____d____
//   0  | a b c |  |_7_6_5_4_3_2_1_0_|_7_..._0_|_7_..._0_|_7_..._0_|...
//   1  | d e f |    0 1    ...    7   8 ... 15 16 ... 23 24 ... 31 ...
//   2  |_g_h_i_|
//
// i.e. out_patch = {a[7:0], b[7:0], c[7:0], d[7:0], ..., i[7:0]}
output reg [V_BITW-1:0]     out_vcnt;
output reg [H_BITW-1:0]     out_hcnt;
// if the center position is (2, 1) in the example above,
// these output coordinates correspond to the pixel <h>

// integer / genvar --------------------------------------------------------
genvar      v, h;

// patch extraction --------------------------------------------------------
reg [BIT_WIDTH-1:0]     patch[0:PATCH_HEIGHT-1][0:PATCH_WIDTH-1];
generate
//enable Buff
reg reg_enable;
always @(posedge clock)
    reg_enable <= enable;
// <delay> modules (FIFO)
for(v = 1; v < PATCH_HEIGHT; v = v + 1) begin: stp_delay_v
    wire [BIT_WIDTH-1:0] delay_out;
    delay
    #( .BIT_WIDTH(BIT_WIDTH), .LATENCY(FRAME_WIDTH - PATCH_WIDTH) )
    dly_0
    (  .clock(clock),      .n_rst(n_rst),   .enable(reg_enable),
    .in_data(patch[v][0]), .out_data(delay_out) );
end
// patch (shift registers)
for(v = 0; v < PATCH_HEIGHT; v = v + 1) begin: stp_patch_v
    for(h = 0; h < PATCH_WIDTH - 1; h = h + 1) begin: stp_patch_h
        always @(posedge clock) begin
            if(reg_enable)begin
                patch[v][h] <= patch[v][h+1];
            end else begin
                patch[v][h] <= patch[v][h];
            end
        end
    end
    if(v == PATCH_HEIGHT - 1) begin
        always @(posedge clock) begin
            if(reg_enable)begin
                patch[v][PATCH_WIDTH-1] <= in_pixel;
            end else begin
                patch[v][PATCH_WIDTH-1] <= patch[v][PATCH_WIDTH-1];
            end
        end
    end
    else begin
        always @(posedge clock)begin
            if(reg_enable)begin
                patch[v][PATCH_WIDTH-1] <= stp_delay_v[v+1].delay_out;
            end else begin
                patch[v][PATCH_WIDTH-1] <= stp_delay_v[v+1].delay_out;
            end
        end
    end
end     
endgenerate

// coordinates adjustment based on the given center position ---------------
wire [V_BITW-1:0]     ctr_vcnt;
wire [H_BITW-1:0]     ctr_hcnt;
//coord_adjuster
//#( .HEIGHT(FRAME_HEIGHT), .WIDTH(FRAME_WIDTH),
//.LATENCY( (PATCH_HEIGHT - 1 - CENTER_V) * FRAME_WIDTH + (PATCH_WIDTH - 1 - CENTER_H) + 1 ) )
//ca_0
//(  .clock(clock), .in_vcnt(in_vcnt), .in_hcnt(in_hcnt),
//.out_vcnt(ctr_vcnt), .out_hcnt(ctr_hcnt)  );
delay
#(
    .BIT_WIDTH(V_BITW),
    .LATENCY( (PATCH_HEIGHT - 1 - CENTER_V) * FRAME_WIDTH + (PATCH_WIDTH - 1 - CENTER_H) + 1 )
)
delay_v
(
    .clock(clock), .n_rst(n_rst),
    .enable(enable), .in_data(in_vcnt), .out_data(ctr_vcnt)
);
delay
#(
    .BIT_WIDTH(H_BITW),
    .LATENCY( (PATCH_HEIGHT - 1 - CENTER_V) * FRAME_WIDTH + (PATCH_WIDTH - 1 - CENTER_H) + 1 )
)
delay_h
(
    .clock(clock), .n_rst(n_rst),
    .enable(enable), .in_data(in_hcnt), .out_data(ctr_hcnt)
);


// padding and output ------------------------------------------------------
generate
for(v = 0; v < PATCH_HEIGHT; v = v + 1) begin: stp_pad_v
    for(h = 0; h < PATCH_WIDTH; h = h + 1) begin: stp_pad_h
        wire [log2(PATCH_HEIGHT)-1:0] tgt_v;
        wire [log2(PATCH_WIDTH)-1:0]  tgt_h;
        if(PADDING == 0) begin
            assign tgt_v = v;
            assign tgt_h = h;
        end
        else begin
            assign tgt_v = (v + ctr_vcnt < CENTER_V) ? CENTER_V - ctr_vcnt :
            (IMAGE_HEIGHT + CENTER_V <= v + ctr_vcnt) ?
            (CENTER_V + IMAGE_HEIGHT - 1) - ctr_vcnt : v;
            assign tgt_h = (h + ctr_hcnt < CENTER_H) ? CENTER_H - ctr_hcnt :
            (IMAGE_WIDTH + CENTER_H <= h + ctr_hcnt) ?
            (CENTER_H + IMAGE_WIDTH - 1) - ctr_hcnt : h;
        end
        always @(posedge clock)
            out_patch[(v * PATCH_WIDTH + h) * BIT_WIDTH +: BIT_WIDTH] <= ((ctr_vcnt < IMAGE_HEIGHT) && (ctr_hcnt < IMAGE_WIDTH)) ? patch[tgt_v][tgt_h] : 0;
    end
end
always @(posedge clock)
    {out_vcnt, out_hcnt} <= {ctr_vcnt, ctr_hcnt}; 
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
