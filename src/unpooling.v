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
// maxpooling kfujita
//

`default_nettype none
`timescale 1ns/1ns
`define UL 3'b000
`define UR 3'b001
`define LL 3'b010
`define LR 3'b011
`define EN_WAIT 3'b100
`define LN_WAIT 3'b101

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
    in_enable,
    in_pixels, in_vcnt,  in_hcnt,
    out_enable,
    out_pixels, out_vcnt, out_hcnt );

// local parameters --------------------------------------------------------
localparam integer FLT_SIZE   = 2;                   // 2x2 max pooling
//localparam integer IN_PIXS    = FLT_SIZE * FLT_SIZE * UNITS;
//localparam integer LATENCY    = 3 + log2(IN_PIXS);  // meccazyuuouyade tyanntosina
//localparam integer LATENCY    = 4;  /////////////////// mecca zisin naiyo!!!!
//localparam integer PATCH_SIZE = FLT_SIZE;
//localparam integer PATCH_BITW = PATCH_SIZE * PATCH_SIZE * FIXED_BITW;
localparam integer V_BITW     = log2(W_HEIGHT);
localparam integer H_BITW     = log2(W_WIDTH);
// LEVEL BUF parameter
// 640 は入力時のWIDTHです。こうしないと正しく遅延できない？
localparam integer FIRST_WIDTH = 640;
localparam integer UppR_BUF    =  1 << LEVEL;
localparam integer LowL_BUF    = (1 << LEVEL) * FIRST_WIDTH;
localparam integer LowR_BUF    = UppR_BUF + LowL_BUF;
localparam integer BUF_BITW    = log2(UppR_BUF);
localparam integer FIRST_WBITW = log2(FIRST_WIDTH);
localparam integer FIRST_TIME  = FIRST_WIDTH*UppR_BUF;
localparam integer FIRST_TIME_WBITW = log2(FIRST_TIME);
// inputs/outputs ----------------------------------------------------------
input wire                             clock, n_rst, in_enable;
input wire [0:FIXED_BITW*UNITS-1]      in_pixels;
input wire [V_BITW-1:0]                in_vcnt;
input wire [H_BITW-1:0]                in_hcnt;
output wire                            out_enable;
output wire [0:FIXED_BITW*UNITS-1]     out_pixels;
output wire [V_BITW:0]                 out_vcnt;    //[log2(V_BITW/2)-1:0] ?
output wire [H_BITW:0]                 out_hcnt;    //[log2(H_BITW/2)-1:0] ?
// -------------------------------------------------------------------------
// genvar      p, v, h, m;

// delay pixels
wire [0:FIXED_BITW*UNITS-1]     uppl_out_pixels;
wire [0:FIXED_BITW*UNITS-1]     uppr_out_pixels;
wire [0:FIXED_BITW*UNITS-1]     lowl_out_pixels;
wire [0:FIXED_BITW*UNITS-1]     lowr_out_pixels;
reg  [0:FIXED_BITW*UNITS-1]     in_pixels_reg;
reg                             in_enable_reg;
reg                             state_enable_reg;

// line_enables
reg [2:0] state_reg;
//reg       prev_vcnt_reg_level;
reg       prev_vcnt_reg_zero;
//wire      prev_vcnt_level;
wire      prev_vcnt_zero;
wire      line_enable;
wire      one_line_enable;
wire      rst_enable;
//assign prev_vcnt_level   = prev_vcnt_reg_level;
//assign line_enable       = (in_vcnt[LEVEL] != prev_vcnt_level);
//assign one_line_enable   = (in_vcnt[0]     != prev_vcnt_zero);
assign prev_vcnt_zero    = prev_vcnt_reg_zero;
assign rst_enable   = (in_vcnt[0]  != prev_vcnt_zero);
always @(posedge clock)begin
    //prev_vcnt_reg_level <= in_vcnt[LEVEL];
    prev_vcnt_reg_zero  <= in_vcnt[0];
end

// state_enable
wire               state_enable;
//reg                state_enable_reg;
reg [BUF_BITW-1:0] state_count;
assign state_enable = (state_count == UppR_BUF - 1);
always @(posedge clock) begin
    if(in_enable)begin
        state_count <= 0;
    end
    else begin
        if(state_count == UppR_BUF - 1)begin
            state_count <= 0;
        end
        else begin
            state_count  <= state_count + 1;
        end
    end
end
//always @(posedge clock)begin
//    state_enable_reg <= state_enable;
//end

wire BEEN_enable;//begin and end
reg [FIRST_WBITW-1:0] one_line_count;
reg [BUF_BITW-1:0] line_count;
assign one_line_enable = (one_line_count == FIRST_WIDTH-1);
assign line_enable = line_count == (1 << LEVEL);
assign BEEN_enable = (LEVEL==0) ? 1 : (line_count==0) || (line_count == (1 << LEVEL));
assign out_enable  = BEEN_enable && state_enable_reg;
always @(posedge clock) begin
    if(rst_enable)begin
        one_line_count <= 0;
        line_count     <= 0;
    end
    else begin
        if(one_line_count == FIRST_WIDTH-1)begin
            one_line_count <= 0;
        end else begin
            one_line_count <= one_line_count + 1;
        end
        if(line_count == UppR_BUF)begin
            line_count <= 0;
        end else if(one_line_enable) begin
            line_count <= line_count + 1;
        end else begin
            line_count <= line_count;
        end
    end
end

reg [H_BITW-1:0]                hcnt_reg;
reg [V_BITW-1:0]                vcnt_reg;
wire [V_BITW-1:0]               uppl_out_vcnt;
wire [H_BITW-1:0]               uppl_out_hcnt;
wire [V_BITW-1:0]               uppr_out_vcnt;
wire [H_BITW-1:0]               uppr_out_hcnt;
wire [V_BITW-1:0]               lowl_out_vcnt;
wire [H_BITW-1:0]               lowl_out_hcnt;
wire [V_BITW-1:0]               lowr_out_vcnt;
wire [H_BITW-1:0]               lowr_out_hcnt;

always @(posedge clock)begin
    in_pixels_reg    <= in_pixels;
    in_enable_reg    <= in_enable;
    state_enable_reg <= state_enable;
    hcnt_reg         <= in_hcnt;
    vcnt_reg         <= in_vcnt;
end
assign uppl_out_pixels = in_pixels_reg;
assign uppl_out_hcnt   = hcnt_reg;
assign uppl_out_vcnt   = vcnt_reg;
//assign out_enable      = LEVEL==0 ? 1'b1 : out_vcnt[LEVEL-1:0]=='b0 && out_hcnt[LEVEL-1:0]=='b0;

delay
#(  .BIT_WIDTH(FIXED_BITW*UNITS),
    .LATENCY(UppR_BUF)
)
delay_UppR
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(in_pixels_reg), .out_data(uppr_out_pixels)
);
delay
#(  .BIT_WIDTH(FIXED_BITW*UNITS),
    .LATENCY(LowL_BUF)
)
delay_LowL
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(in_pixels_reg), .out_data(lowl_out_pixels)
);
delay
#(  .BIT_WIDTH(FIXED_BITW*UNITS),
    .LATENCY(LowR_BUF)
)
delay_LowR
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(in_pixels_reg), .out_data(lowr_out_pixels)
);

delay
#(  .BIT_WIDTH(V_BITW),
    .LATENCY(UppR_BUF)
)
delay_UppR_V
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(vcnt_reg), .out_data(uppr_out_vcnt)
);
delay
#(  .BIT_WIDTH(H_BITW),
    .LATENCY(UppR_BUF)
)
delay_UppR_H
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(hcnt_reg), .out_data(uppr_out_hcnt)
);

delay
#(  .BIT_WIDTH(V_BITW),
    .LATENCY(LowL_BUF)
)
delay_LowL_V
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(vcnt_reg), .out_data(lowl_out_vcnt)
);
delay
#(  .BIT_WIDTH(H_BITW),
    .LATENCY(LowL_BUF)
)
delay_LowL_H
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(hcnt_reg), .out_data(lowl_out_hcnt)
);

delay
#(  .BIT_WIDTH(V_BITW),
    .LATENCY(LowR_BUF)
)
delay_LowR_V
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(vcnt_reg), .out_data(lowr_out_vcnt)
);
delay
#(  .BIT_WIDTH(H_BITW),
    .LATENCY(LowR_BUF)
)
delay_LowR_H
(   .clock(clock),  .n_rst(n_rst),
    .enable(1),
    .in_data(hcnt_reg), .out_data(lowr_out_hcnt)
);

always @(posedge clock or negedge n_rst) begin
    if(!n_rst) begin
        state_reg <= `EN_WAIT;
    end
    else begin
        case(state_reg)
            `UL:begin
                if(state_enable)begin
                    state_reg <= `UR;
                end
                else state_reg <= `UL;
            end
            `UR:begin
                if(one_line_enable)begin
                    if(LEVEL==0)begin
                        state_reg <= `LL;
                    end else begin
                        state_reg <= `LN_WAIT;
                    end
                end
                else if(in_enable)begin
                    state_reg <= `UL;
                end
                else state_reg <= `UR;
            end
            `LN_WAIT:begin
                if(line_enable)begin
                    state_reg <= `LL;
                end else begin
                    state_reg <= `LN_WAIT;
                end
            end
            `LL:begin
                if(state_enable)begin
                    state_reg <= `LR;
                end
                else state_reg <= `LL;
            end
            `LR:begin
                if(one_line_enable)begin
                    if(LEVEL==0)begin
                        state_reg <= `UL;
                    end else begin
                        state_reg <= `EN_WAIT;
                    end
                end
                else if(state_enable)begin
                    state_reg <= `LL;
                end
                else state_reg <= `LR;
            end
            `EN_WAIT:begin
                if(in_enable)begin
                    state_reg <= `UL;
                end else begin
                    state_reg <= `EN_WAIT;
                end
            end
            default:begin
                state_reg <= `EN_WAIT;
            end
        endcase
    end
end

assign out_pixels = choiceOUT(state_reg,uppl_out_pixels,uppr_out_pixels,lowl_out_pixels,lowr_out_pixels);

function [0:FIXED_BITW*UNITS-1] choiceOUT;
    input [2:0] state_reg;
    input [0:FIXED_BITW*UNITS-1] uppl_out_pixels;
    input [0:FIXED_BITW*UNITS-1] uppr_out_pixels;
    input [0:FIXED_BITW*UNITS-1] lowl_out_pixels;
    input [0:FIXED_BITW*UNITS-1] lowr_out_pixels;
    begin
        case(state_reg)
        `UL: choiceOUT = uppl_out_pixels;
        `UR: choiceOUT = uppr_out_pixels;
        `LL: choiceOUT = lowl_out_pixels;
        `LR: choiceOUT = lowr_out_pixels;
        default: choiceOUT = 'bx;
        endcase
    end
endfunction

assign out_vcnt = choiceV(state_reg,uppl_out_vcnt,uppr_out_vcnt,lowl_out_vcnt,lowr_out_vcnt);
function [V_BITW:0] choiceV;
    input [2:0] state_reg;
    input [V_BITW-1:0] uppl_out_vcnt;
    input [V_BITW-1:0] uppr_out_vcnt;
    input [V_BITW-1:0] lowl_out_vcnt;
    input [V_BITW-1:0] lowr_out_vcnt;
    begin
        case(state_reg)
        `UL: choiceV = {uppl_out_vcnt,1'b0};
        `UR: choiceV = {uppr_out_vcnt,1'b0};
        `LL: choiceV = {lowl_out_vcnt,1'b1};
        `LR: choiceV = {lowr_out_vcnt,1'b1};
        default: choiceV = 'bx;
        endcase
    end
endfunction

assign out_hcnt = choiceH(state_reg,uppl_out_hcnt,uppr_out_hcnt,lowl_out_hcnt,lowr_out_hcnt);
function [H_BITW:0] choiceH;
    input [2:0] state_reg;
    input [H_BITW-1:0] uppl_out_hcnt;
    input [H_BITW-1:0] uppr_out_hcnt;
    input [H_BITW-1:0] lowl_out_hcnt;
    input [H_BITW-1:0] lowr_out_hcnt;
    begin
        case(state_reg)
        `UL: choiceH = {uppl_out_hcnt,1'b0};
        `UR: choiceH = {uppr_out_hcnt,1'b1};
        `LL: choiceH = {lowl_out_hcnt,1'b0};
        `LR: choiceH = {lowr_out_hcnt,1'b1};
        default: choiceH = 'bx;
        endcase
    end
endfunction

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
