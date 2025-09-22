#!/usr/bin/env python


'''
Written by Ilariat21 (06/13/2025)

Usage: python gen_input.py <num_sinks>
example: python gen_input.py 1000
'''

import sys
import numpy as np

if len(sys.argv)<2:
    sys.exit("Usage: python {0} <num_sinks>".format(sys.argv[0]))

num_sinks = sys.argv[1]        # sink count

# Sink count is the only required parameter, others can be defined here
chip_size = int(num_sinks)*2   # chip width (= chip height)
unit_R = 1.0                   # unit resistance
unit_C = 0.5                   # unit capacitance
max_cap = 10.0                 # max capacitance constraint
buffer_input_cap = 2.0         # buffer input capacitance
buffer_fixed_delay = 4.0       # buffer delay
clk_x, clk_y = 0, 10           # clock input pin location (x, y)

filename = f"input_{num_sinks}.txt"
f = open(filename, 'w')

f.write(f"{num_sinks} {chip_size} {unit_R} {unit_C} {max_cap} {buffer_input_cap} {buffer_fixed_delay}\n")

f.write(f"{clk_x} {clk_y}\n")

sink_dict = {}
sink_count = 0
while sink_count < int(num_sinks):
    r = np.random.randint(low = 1, high=chip_size)
    c = np.random.randint(low = 1, high=chip_size)
    cap = np.random.uniform(low = 0.1, high = 5.0)
    if (r, c) not in sink_dict.keys():
        sink_dict[r, c] = cap
        f.write(f"{r} {c} {cap}\n")
        sink_count += 1
