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
// kfujita
//  - stream_patch support padding (12_17, 2019)
//  - patch_size 5->3 delete flip (12_18,2019)
//-----------------------------------------------------------------------------
`default_nettype none
`timescale 1ns/1ns

module layer
#(  parameter integer WIDTH         = -1,
    parameter integer HEIGHT        = -1,
    parameter integer W_WIDTH       = -1,
    parameter integer W_HEIGHT      = -1,
    parameter integer UINT_BITW     = -1, // cb/cr bit width (non used)
    parameter integer FRAC_BITW     = -1, // fractional part bit width
    parameter integer PREV_INT_BITW = -1, // | integer part bit width
    parameter integer INT_BITW      = -1, // |
    parameter integer PREV_UNITS    = -1, // | # of units
    parameter integer UNITS         = -1, // |
    parameter integer FLT_SIZE      = -1, // filter size
    parameter [0:(INT_BITW+FRAC_BITW)*PREV_UNITS*UNITS*FLT_SIZE*FLT_SIZE-1] FLT = 0,
    parameter [0:(INT_BITW+FRAC_BITW*2)*UNITS-1] BIAS = 0,
    parameter integer LEVEL         = -1 )
(   clock,      n_rst,
    in_enable,
    in_pixels, in_vcnt,  in_hcnt,
    out_enable,
    out_pixels, out_vcnt, out_hcnt );

// local parameters --------------------------------------------------------
localparam integer PREV_BITW  = PREV_INT_BITW + FRAC_BITW;
localparam integer FIXED_BITW = INT_BITW      + FRAC_BITW;
localparam integer RES_BITW   = INT_BITW      + FRAC_BITW * 2;
localparam integer IN_PIXS    = FLT_SIZE * FLT_SIZE * PREV_UNITS;
localparam integer FLT_PIXS   = IN_PIXS  * UNITS;
//localparam integer LATENCY    = 4 + log2(IN_PIXS);
localparam integer LATENCY    = 4 + log2(IN_PIXS);
//localparam integer PATCH_SIZE = FLT_SIZE * 2 - 1;
localparam integer PATCH_SIZE = FLT_SIZE;
localparam integer PATCH_BITW = PATCH_SIZE * PATCH_SIZE * FIXED_BITW;
localparam integer V_BITW     = log2(W_HEIGHT);
localparam integer H_BITW     = log2(W_WIDTH);

localparam integer CENTER_H   = PATCH_SIZE - 1;
localparam integer CENTER_V   = PATCH_SIZE - 1;

// inputs/outputs ----------------------------------------------------------
input wire                             clock, n_rst, in_enable;
input wire [0:PREV_BITW*PREV_UNITS-1]  in_pixels;
input wire [V_BITW-1:0]                in_vcnt;
input wire [H_BITW-1:0]                in_hcnt;
output wire                            out_enable;
output wire [0:FIXED_BITW*UNITS-1]     out_pixels;
output wire [V_BITW-1:0]               out_vcnt;
output wire [H_BITW-1:0]               out_hcnt;

// -------------------------------------------------------------------------
genvar      p, v, h, m;

// buffering and generating input images -----------------------------------
generate

// patch extraction --------------------------------------------------
reg [H_BITW-1:0]             stp_hcnt;
reg [V_BITW-1:0]             stp_vcnt;
reg [0:FIXED_BITW*IN_PIXS-1] flip_pixels;
wire                  stp_enable;

for(p = 0; p < PREV_UNITS; p = p + 1) begin : ly_patch

    // bit extension -----------------------------------------------
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

    // stream patch ------------------------------------------------
    wire [0:PATCH_BITW-1] stp_patch;
    wire [H_BITW-1:0]     stp_hcnt_w;
    wire [V_BITW-1:0]     stp_vcnt_w;
    stream_patch
    #(  .BIT_WIDTH(FIXED_BITW),
        .IMAGE_HEIGHT(HEIGHT),     .IMAGE_WIDTH(WIDTH),
        .FRAME_HEIGHT(W_HEIGHT),   .FRAME_WIDTH(W_WIDTH),
        .PATCH_HEIGHT(PATCH_SIZE), .PATCH_WIDTH(PATCH_SIZE),
        .CENTER_V(PATCH_SIZE / 2),   .CENTER_H(PATCH_SIZE / 2),
        .PADDING(1), .LEVEL(LEVEL) )
    stp_0
    (   .clock(clock),         .n_rst(n_rst),
        .enable(in_enable),
        .in_pixel(new_pixel),
        .in_hcnt(in_hcnt),     .in_vcnt(in_vcnt),
        .out_patch(stp_patch),
        .out_hcnt(stp_hcnt_w), .out_vcnt(stp_vcnt_w),
        .out_enable(stp_enable)  );

    // discrete extraction -----------------------------------------
    reg [0:FIXED_BITW-1]  stp_img [0:FLT_SIZE-1][0:FLT_SIZE-1];
    for(v = 0; v < FLT_SIZE; v = v + 1) begin : ext_v
        for(h = 0; h < FLT_SIZE; h = h + 1) begin : ext_h
            always @(posedge clock)
                stp_img[v][h] <= stp_patch[(v*PATCH_SIZE + h)*FIXED_BITW +: FIXED_BITW]; // change
        end
    end

    if(p == 0) begin
        always @(posedge clock)
            {stp_hcnt, stp_vcnt} <= {stp_hcnt_w, stp_vcnt_w};
    end

    // applies the flip corresponding to the coordinates -----------
    ////////////////////////////////////////////////////////////////
    // change not flip (12_19, 2019 kfujita)
    ////////////////////////////////////////////////////////////////
    //flip_pixels[((p * FLT_SIZE + v) * FLT_SIZE + h) * FIXED_BITW +: FIXED_BITW] 
    // <= stp_patch[(v*PATCH_SIZE + h)*FIXED_BITW +: FIXED_BITW]; //?????????????????????????????????
    ////////////////////////////////////////////////////////////////
    for(v = 0; v < FLT_SIZE; v = v + 1) begin : flip_v
        for(h = 0; h < FLT_SIZE; h = h + 1) begin : flip_h

            //wire [log2(FLT_SIZE)-1:0] v_pos, h_pos;
            //assign v_pos = (stp_vcnt[0] == 0) ? v : FLT_SIZE - 1 - v;
            //assign h_pos = (stp_hcnt[0] == 0) ? h : FLT_SIZE - 1 - h;

            always @(posedge clock)
                flip_pixels[((p * FLT_SIZE + v) * FLT_SIZE + h) * FIXED_BITW +: FIXED_BITW] <= stp_img[v][h];
        end
    end

end

// forward propagation -----------------------------------------------
wire [0:RES_BITW*UNITS-1]   fwd_out_pixels;
forward
#(  .IN_PIXS(IN_PIXS),   .OUT_PIXS(UNITS),
    .INT_BITW(INT_BITW), .FRAC_BITW(FRAC_BITW),
    .FLT(FLT),           .BIAS(BIAS) )
fwd_0
(   .clock(clock),
    .in_pixels(flip_pixels),
    .out_pixels(fwd_out_pixels) );

// applies Leaky ReLU and nearest rounding ---------------------------
for(p = 0; p < UNITS; p = p + 1) begin : ly_out_pixel

    // applies Leaky ReLU and nearest rounding ---------------------
    wire signed [RES_BITW-1:0]  val, activ;
    reg signed [FIXED_BITW-1:0] result;

    assign val    = fwd_out_pixels[RES_BITW * p +: RES_BITW];
    assign activ  = (0 <= val) ? val : $signed(val + 2) >>> 2;
    always @(posedge clock)
        result <= $signed((activ >>> (FRAC_BITW - 1)) + 1) >>> 1;

    // assigns result ----------------------------------------------
    assign out_pixels[p * FIXED_BITW +: FIXED_BITW] = result;

end
endgenerate

// coordinates adjustment --------------------------------------------------
coord_adjuster
#(  .HEIGHT(W_HEIGHT), .WIDTH(W_WIDTH), .LATENCY(LATENCY) )
ca_1
(   .clock(clock), .in_vcnt(stp_vcnt), .in_hcnt(stp_hcnt),
    .out_vcnt(out_vcnt), .out_hcnt(out_hcnt) );
// coord adjuster change delay
//delay
//#(  .BIT_WIDTH(H_BITW),
//    .LATENCY(LATENCY)
//)
//delay_h
//(   .clock(clock),  .n_rst(n_rst),
//    .enable(1),
//    .in_data(stp_hcnt), .out_data(out_hcnt)
//);
//delay
//#(  .BIT_WIDTH(V_BITW),
//    .LATENCY(LATENCY)
//)
//delay_v
//(   .clock(clock),  .n_rst(n_rst),
//    .enable(1),
//    .in_data(stp_vcnt), .out_data(out_vcnt)
//);

// delay for cb/cr channels ------------------------------------------------
// delay for enable signal
delay
#(  .BIT_WIDTH(1),
    .LATENCY((LATENCY + 2))
)
delay_inst
(
    .clock(clock),  .n_rst(n_rst),
    .enable(1'b1),
    .in_data(stp_enable), .out_data(out_enable)
);
// delay .LATENCY( (((PATCH_SIZE - 1 - CENTER_V) * W_WIDTH + (PATCH_SIZE - 1 - CENTER_H) + 1 ) + 1) + 1 + LATENCY  )
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
