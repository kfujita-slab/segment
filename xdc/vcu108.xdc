# Define and contraint system clock

#create_clock -name clk_1 -period 30  [get_ports clk_1]
#set_propagated_clock clk_1
#set_property PACKAGE_PIN BC9         [get_ports clk_1]
#set_property IOSTANDARD LVCMOS18     [get_ports clk_1]

#create_clock -name clk_4 -period 7.5 [get_ports clk_4]
#set_propagated_clock clk_4
#set_property PACKAGE_PIN G22         [get_ports clk_4]
#set_property IOSTANDARD LVCMOS18     [get_ports clk_4]

create_clock -name clock -period 10.0 [get_ports clock]
set_propagated_clock clock
set_property PACKAGE_PIN G22         [get_ports clock]
set_property IOSTANDARD LVCMOS18     [get_ports clock]

set_property PACKAGE_PIN AU21        [get_ports "n_rst"]
set_property IOSTANDARD  LVCMOS18    [get_ports "n_rst"]

set_property PACKAGE_PIN J37         [get_ports "in_y[0]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[0]"]
set_property PACKAGE_PIN H40         [get_ports "in_y[1]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[1]"]
set_property PACKAGE_PIN F38         [get_ports "in_y[2]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[2]"]
set_property PACKAGE_PIN H39         [get_ports "in_y[3]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[3]"]
set_property PACKAGE_PIN K37         [get_ports "in_y[4]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[4]"]
set_property PACKAGE_PIN G40         [get_ports "in_y[5]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[5]"]
set_property PACKAGE_PIN F39         [get_ports "in_y[6]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[6]"]
set_property PACKAGE_PIN F40         [get_ports "in_y[7]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[7]"]

set_property PACKAGE_PIN F36         [get_ports "in_y[8]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[8]"]
set_property PACKAGE_PIN J36         [get_ports "in_y[9]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[9]"]
set_property PACKAGE_PIN F35         [get_ports "in_y[10]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[10]"]
set_property PACKAGE_PIN J35         [get_ports "in_y[11]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[11]"]
set_property PACKAGE_PIN G37         [get_ports "in_y[12]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[12]"]
set_property PACKAGE_PIN H35         [get_ports "in_y[13]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[13]"]
set_property PACKAGE_PIN G36         [get_ports "in_y[14]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[14]"]
set_property PACKAGE_PIN H37         [get_ports "in_y[15]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[15]"]

set_property PACKAGE_PIN J39         [get_ports "in_y[16]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[16]"]
set_property PACKAGE_PIN F34         [get_ports "in_y[17]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[17]"]
set_property PACKAGE_PIN C39         [get_ports "in_y[18]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[18]"]
set_property PACKAGE_PIN A38         [get_ports "in_y[19]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[19]"]
set_property PACKAGE_PIN B40         [get_ports "in_y[20]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[20]"]
set_property PACKAGE_PIN D40         [get_ports "in_y[21]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[21]"]
set_property PACKAGE_PIN E38         [get_ports "in_y[22]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[22]"]
set_property PACKAGE_PIN B38         [get_ports "in_y[23]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_y[23]"]

set_property PACKAGE_PIN E37         [get_ports "in_vcnt[0]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[0]"]
set_property PACKAGE_PIN C40         [get_ports "in_vcnt[1]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[1]"]
set_property PACKAGE_PIN C34         [get_ports "in_vcnt[2]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[2]"]
set_property PACKAGE_PIN A34         [get_ports "in_vcnt[3]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[3]"]
set_property PACKAGE_PIN D34         [get_ports "in_vcnt[4]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[4]"]
set_property PACKAGE_PIN A35         [get_ports "in_vcnt[5]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[5]"]
set_property PACKAGE_PIN A36         [get_ports "in_vcnt[6]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[6]"]
set_property PACKAGE_PIN C35         [get_ports "in_vcnt[7]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[7]"]
set_property PACKAGE_PIN B35         [get_ports "in_vcnt[8]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[8]"]
#set_property PACKAGE_PIN D35         [get_ports "in_vcnt[9]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "in_vcnt[9]"]

set_property PACKAGE_PIN E39         [get_ports "in_hcnt[0]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[0]"]
set_property PACKAGE_PIN D37         [get_ports "in_hcnt[1]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[1]"]
set_property PACKAGE_PIN N27         [get_ports "in_hcnt[2]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[2]"]
set_property PACKAGE_PIN R27         [get_ports "in_hcnt[3]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[3]"]
set_property PACKAGE_PIN N24         [get_ports "in_hcnt[4]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[4]"]
set_property PACKAGE_PIN R24         [get_ports "in_hcnt[5]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[5]"]
set_property PACKAGE_PIN P24         [get_ports "in_hcnt[6]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[6]"]
set_property PACKAGE_PIN P26         [get_ports "in_hcnt[7]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[7]"]
set_property PACKAGE_PIN P27         [get_ports "in_hcnt[8]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[8]"]
set_property PACKAGE_PIN T24         [get_ports "in_hcnt[9]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "in_hcnt[9]"]

set_property PACKAGE_PIN K27         [get_ports "out_y[0]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_y[0]"]
set_property PACKAGE_PIN L26         [get_ports "out_y[1]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_y[1]"]
#set_property PACKAGE_PIN J27         [get_ports "out_r[2]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_r[2]"]
#set_property PACKAGE_PIN K28         [get_ports "out_r[3]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_r[3]"]
#set_property PACKAGE_PIN K26         [get_ports "out_r[4]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_r[4]"]
#set_property PACKAGE_PIN M25         [get_ports "out_r[5]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_r[5]"]
#set_property PACKAGE_PIN J26         [get_ports "out_r[6]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_r[6]"]
#set_property PACKAGE_PIN L28         [get_ports "out_r[7]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_r[7]"]
#
#set_property PACKAGE_PIN T26         [get_ports "out_g[0]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[0]"]
#set_property PACKAGE_PIN M27         [get_ports "out_g[1]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[1]"]
#set_property PACKAGE_PIN E27         [get_ports "out_g[2]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[2]"]
#set_property PACKAGE_PIN E28         [get_ports "out_g[3]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[3]"]
#set_property PACKAGE_PIN E26         [get_ports "out_g[4]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[4]"]
#set_property PACKAGE_PIN H27         [get_ports "out_g[5]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[5]"]
#set_property PACKAGE_PIN F25         [get_ports "out_g[6]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[6]"]
#set_property PACKAGE_PIN F28         [get_ports "out_g[7]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_g[7]"]
#
#set_property PACKAGE_PIN G25         [get_ports "out_b[0]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[0]"]
#set_property PACKAGE_PIN G27         [get_ports "out_b[1]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[1]"]
#set_property PACKAGE_PIN B28         [get_ports "out_b[2]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[2]"]
#set_property PACKAGE_PIN A28         [get_ports "out_b[3]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[3]"]
#set_property PACKAGE_PIN B25         [get_ports "out_b[4]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[4]"]
#set_property PACKAGE_PIN B27         [get_ports "out_b[5]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[5]"]
#set_property PACKAGE_PIN D25         [get_ports "out_b[6]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[6]"]
#set_property PACKAGE_PIN C27         [get_ports "out_b[7]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_b[7]"]

set_property PACKAGE_PIN C25         [get_ports "out_vcnt[0]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[0]"]
set_property PACKAGE_PIN D26         [get_ports "out_vcnt[1]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[1]"]
set_property PACKAGE_PIN G26         [get_ports "out_vcnt[2]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[2]"]
set_property PACKAGE_PIN D27         [get_ports "out_vcnt[3]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[3]"]
set_property PACKAGE_PIN N29         [get_ports "out_vcnt[4]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[4]"]
set_property PACKAGE_PIN M31         [get_ports "out_vcnt[5]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[5]"]
set_property PACKAGE_PIN P29         [get_ports "out_vcnt[6]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[6]"]
set_property PACKAGE_PIN L29         [get_ports "out_vcnt[7]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[7]"]
set_property PACKAGE_PIN P30         [get_ports "out_vcnt[8]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[8]"]
#set_property PACKAGE_PIN N28         [get_ports "out_vcnt[9]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[9]"]
#set_property PACKAGE_PIN L31         [get_ports "out_vcnt[10]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_vcnt[10]"]

set_property PACKAGE_PIN L30         [get_ports "out_hcnt[0]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[0]"]
set_property PACKAGE_PIN H30         [get_ports "out_hcnt[1]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[1]"]
set_property PACKAGE_PIN J32         [get_ports "out_hcnt[2]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[2]"]
set_property PACKAGE_PIN H29         [get_ports "out_hcnt[3]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[3]"]
set_property PACKAGE_PIN H32         [get_ports "out_hcnt[4]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[4]"]
set_property PACKAGE_PIN J29         [get_ports "out_hcnt[5]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[5]"]
set_property PACKAGE_PIN K32         [get_ports "out_hcnt[6]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[6]"]
set_property PACKAGE_PIN J30         [get_ports "out_hcnt[7]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[7]"]
set_property PACKAGE_PIN G32         [get_ports "out_hcnt[8]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[8]"]
set_property PACKAGE_PIN R28         [get_ports "out_hcnt[9]"]
set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[9]"]
#set_property PACKAGE_PIN K31         [get_ports "out_hcnt[10]"]
#set_property IOSTANDARD  POD12_DCI   [get_ports "out_hcnt[10]"]
