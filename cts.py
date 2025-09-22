#!/usr/bin/env python

'''
Written by Uegoogin (06/13/2025)

Usage: python cts20235557.py input_file visualize_output[True/False (default: False)]
example: python cts20235557.py testcases/input1.txt True
'''

import sys
import math
import time
import re
import matplotlib.pyplot as plt


def make_node(x, y, cap, type, id=None, delay=0, left=None, right=None):
    temp = {}
    temp['x'] = x
    temp['y'] = y
    temp['cap'] = cap
    temp['type'] = type
    if temp['type'] == 'SINK':
        temp['id'] = f'sink{id}'
    elif temp['type'] == 'BUFFER':
        temp['id'] = f"B{id}"
    elif temp['type'] == 'BRANCH':
        temp['id'] = f'P{id}'
    elif temp['type'] == 'SOURCE':
        temp['id'] = 'source'
    temp['left'] = left
    temp['right'] = right
    temp['delay'] = delay
    return temp


def readfile(input_dir):
    f = open(input_dir, 'r')
    lines = f.readlines()

    header = {}
    gen_info = lines[0].split()
    
    header['num_sinks'] = int(gen_info[0])             # sink count
    header['chip_size'] = int(gen_info[1])             # chip width
    header['unit_R'] = float(gen_info[2])              # unit resistance
    header['unit_C'] = float(gen_info[3])              # unit capacitance
    header['max_cap'] = float(gen_info[4])             # max capacitance constraint
    header['buffer_input_cap'] = float(gen_info[5])    # buffer input capacitance
    header['buffer_fixed_del'] = float(gen_info[6])    # buffer delay

    clock_pin = [int(lines[1].split()[0]), int(lines[1].split()[1])]
    print(f'''Input info:\t Sink count = {header['num_sinks']}\t Chip width (=chip height) = {header['chip_size']}\t
          Unit resistance = {header['unit_R']}\t Unit capacitance = {header['unit_C']}\t 
          Max capacitance constraint = {header['max_cap']}\t Buffer input capacitance = {header['buffer_input_cap']}\t
          Buffer delay = {header['buffer_fixed_del']}\t Clock pin coordinates : {clock_pin}\n''')

    node_list = {}           # list of all nodes
    unused_nodes = {}        # list of unmatched nodes

    for i in range(2, len(lines)):
        id, x, y, cap = i-1, int(lines[i].split()[0]), int(lines[i].split()[1]), float(lines[i].split()[2])
        node_list[f'sink{id}'] = make_node(x, y, cap, 'SINK', id)
        unused_nodes[f'sink{id}'] = f'sink{id}'

    if len(node_list)!=header['num_sinks']:
        print(f"Warning! Number of sinks does not match! {len(node_list)}!={header['num_sinks']}\n")
    
    return header, clock_pin, node_list, unused_nodes


def manh_dist(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])


def greedy_pairing(nodes, unused_nodes):
    matched = set()
    pairs = []
    temp = {}
   
    for i in unused_nodes.keys():
        temp[unused_nodes[i]] = nodes[unused_nodes[i]]

    temp = dict(sorted(temp.items(), key=lambda item: (item[1]['x'], item[1]['y'])))
    
    items = list(temp.items())
    for i, (keyi, valuei) in enumerate(items[:len(temp)-1]):
        if keyi in matched:
            continue

        closest_key = None
        min_distance = float('inf')

        for j, (keyj, valuej) in enumerate(items[i+1:]):
            if keyj in matched:
                continue

            pt1 = [valuei['x'], valuei['y']]
            pt2 = [valuej['x'], valuej['y']]
            dist = manh_dist(pt1, pt2)

            if dist < min_distance:
                min_distance = dist
                closest_key = keyj
                closest_item = valuej
            
        if closest_key != None:
            d1, d2 = {}, {}
            d1[keyi] = valuei
            d2[closest_key] = closest_item
            pairs.append([d1, d2])
            matched.add(keyi)
            matched.add(closest_key)
            del unused_nodes[keyi]
            del unused_nodes[closest_key]

    return pairs, unused_nodes


def manh_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def newpt(p1, p2, d, guide, existing_coords):
    from collections import deque

    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # Sign directions
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    # Option 1: move x first
    move_x1 = min(abs(dx), d)
    move_y1 = d - move_x1
    x1_new = x1 + move_x1 * sign_x
    y1_new = y1 + move_y1 * sign_y
    point1 = [x1_new, y1_new]
    dist1 = manh_dist(point1, guide)

    # Option 2: move y first
    move_y2 = min(abs(dy), d)
    move_x2 = d - move_y2
    x2_new = x1 + move_x2 * sign_x
    y2_new = y1 + move_y2 * sign_y
    point2 = [x2_new, y2_new]
    dist2 = manh_dist(point2, guide)

    # Choose better option
    chosen = point1 if dist1 <= dist2 else point2

    # If point is occupied, try nearby points using BFS
    if tuple(chosen) in existing_coords:
        visited = set()
        queue = deque()
        queue.append((chosen[0], chosen[1]))

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) not in existing_coords:
                return [cx, cy]
            visited.add((cx, cy))

            # Try all 4 directions
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited:
                    queue.append((nx, ny))

    return chosen

def tapping_point(k1, k2, p1, p2, coords):
    global unused_nodes, node_list, branch_id, buff_id, xses, yses, clock_pin

    pt1 = [p1['x'], p1['y']]
    pt2 = [p2['x'], p2['y']]
    L = manh_dist(pt1, pt2)
    cap1, cap2 = p1['cap'], p2['cap']
    delay1, delay2 = p1['delay'], p2['delay']
    unit_C, unit_R = header['unit_C'], header['unit_R']

    # Calculate the tapping point distance from point 1
    x = int((unit_C*L**2/2+cap2*L+(delay2-delay1)/unit_R)/(cap1+cap2+unit_C*L))

    pt = newpt(pt1, pt2, x, clock_pin, coords)
    cap = unit_C*L + cap1 + cap2
    type = 'BRANCH'
    if cap > header['max_cap']:  # If capacitance of the new node is over max cap, then place buffer
        cap = header['buffer_input_cap']
        delay = header['buffer_fixed_del']
        type = 'BUFFER'
        id = buff_id
        nm = f'B{id}'
        buff_id += 1
    else:                        # else place branch
        id = branch_id
        nm = f'P{id}'
        branch_id += 1
        delay = max(unit_R*x*(unit_C*x/2+cap1)+delay1, unit_R*(L-x)*(unit_C*(L-x)/2+cap2)+delay2)
    edges.append([pt, pt1, nm, k1])
    edges.append([pt, pt2, nm, k2])

    node_list[nm] = make_node(pt[0], pt[1], cap, type, id, delay, k1, k2)
    unused_nodes[nm] = nm

def manh_segment(edges, xses, yses):
    segments = []
    for i in range(len(edges)):
        pt1, pt2, k1, k2 = edges[i]
        if pt1[0] == pt2[0] or pt1[1] == pt2[1]:
            segments.append([pt1, pt2, k1, k2])
            xses.append([pt1[0], pt2[0]])
            yses.append([pt1[1], pt2[1]])
        else:
            segments.append([[pt1[0], pt1[1]], [pt2[0], pt1[1]], k1, k2])
            segments.append([[pt2[0], pt1[1]], [pt2[0], pt2[1]], k1, k2])
            xses.append([pt1[0], pt2[0]])
            xses.append([pt2[0], pt2[0]])
            yses.append([pt1[1], pt1[1]])
            yses.append([pt1[1], pt2[1]])
    
    return segments, xses, yses


def visualize(node_list, xses, yses, filename, iter=None):
    x_vals = [node_list[i]['x'] for i in list(node_list.keys()) if 'sink' in i]
    y_vals = [node_list[i]['y'] for i in list(node_list.keys()) if 'sink' in i]

    x_valsb = [node_list[i]['x'] for i in list(node_list.keys()) if 'B' in i]
    y_valsb = [node_list[i]['y'] for i in list(node_list.keys()) if 'B' in i]

    x_valsr = [node_list[i]['x'] for i in list(node_list.keys()) if 'P' in i]
    y_valsr = [node_list[i]['y'] for i in list(node_list.keys()) if 'P' in i]

    plt.figure(figsize=(10,10))
    
    plt.scatter(x_vals, y_vals, c='blue', s=900/header['num_sinks'], label='sinks')
    plt.scatter(x_valsb, y_valsb, c='green', s=500/header['num_sinks'], label='buffers')
    plt.scatter(x_valsr, y_valsr, c='brown', s=500/header['num_sinks'], label='branches')
    plt.scatter(*clock_pin, c='red', s=1000/header['num_sinks'], label='Input Clock Pin')

    plt.xlim(0, header['chip_size'])
    plt.ylim(0, header['chip_size'])

    for i in range(len(xses)):
        plt.plot(xses[i], yses[i], 'b-', linewidth=(100/header['num_sinks'])**0.5/2)

    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    if iter==None:
        plt.savefig(f'{filename}.png')
    else:
        plt.savefig(f'{filename}_{iter}.png')
    plt.show()


#########################################################################################
####################################### MAIN FUNC #######################################
#########################################################################################

start = time.time()

if len(sys.argv) < 2:
    sys.exit("Usage: python {0} input_file visualize_output[True/False (default: False)]".format(sys.argv[0]))

input_dir = sys.argv[1]

if len(sys.argv)==3:
    vis_out = sys.argv[2]
else:
    vis_out = False

if '/' in input_dir:
    infn = input_dir.split("/")[-1].split(".")[0]
    output_file = f"{infn}_output.txt"
else:
    infn = input_dir.split(".")[0]
    output_file = f"{infn}_output.txt" 

out = open(output_file, 'w')

xses = []       # List of lines on x axis
yses = []       # List of lines on y axis
edges = []      # List of all edges (src, dst)
buff_id = 1
branch_id = 1

xses1 = []
yses1 = []

header, clock_pin, node_list, unused_nodes = readfile(input_dir)

num_steps = math.ceil(math.log2(header['num_sinks']))

itr = 1

idx = 1
while len(unused_nodes)>1:
    print(f"Processing iteration: {itr}/{num_steps}")
    itr+=1
    idx += 1
    pairs, unused_nodes = greedy_pairing(node_list, unused_nodes)
    for pair in pairs:
        coords = [(node_list[i]['x'], node_list[i]['y']) for i in list(node_list.keys())]
        key1, key2 = list(pair[0].keys())[0], list(pair[1].keys())[0]
        val1, val2 = pair[0][key1], pair[1][key2]
        tapping_point(key1, key2, val1, val2, coords)

def extract_number(key):
    match = re.search(r'\d+', key)
    return int(match.group()) if match else float('inf')

# Sord node list according to node ids
node_list = dict(sorted(node_list.items(), key=lambda item: (item[0][0], extract_number(item[0]))))

# Connect clock pin with first tapping point
k1 = list(unused_nodes.keys())[0]
edges.append([clock_pin, [node_list[k1]['x'], node_list[k1]['y']], 'source', k1])

segments, xses, yses = manh_segment(edges, xses, yses)

# Calculate total wire length
total_wl = 0
for i in range(len(edges)):
    pt1, pt2 = edges[i][0], edges[i][1]
    total_wl += manh_dist(pt1, pt2)

# Visualize clock tree
if vis_out:
    visualize(node_list, xses, yses, f'vis_{infn}')


#### Writing the output ####

print("\nSaving the output...")
out.write(f"NODES {len(node_list)+1}\n") # +1 because source is not in the node list
out.write(f"source {clock_pin[0]} {clock_pin[1]} SOURCE\n")

keys = list(node_list.keys())
for i in range(len(keys)):
    tempd = node_list[keys[i]]
    x, y, cap, type, id = tempd['x'], tempd['y'], tempd['cap'], tempd['type'], tempd['id']
    if type != 'SINK':
        out.write(f"{id} {x} {y} {type}\n")
    elif type == 'SINK':
        out.write(f"{id} {x} {y} {cap} {type}\n")

out.write(f"\nSEGMENTS {len(segments)}\n")

for i in range(len(segments)):
    out.write(f"{segments[i][2]} {segments[i][3]} {segments[i][0][0]} {segments[i][0][1]} {segments[i][1][0]} {segments[i][1][1]}\n")

print(f"Done! Output is written and saved to '{output_file}'\n")

num_bufs = 0
for i in list(node_list.keys()):
    if "B" in i:
        num_bufs+=1

print("Execution info:")
print("Number of buffers =", num_bufs)
print("Total_wl =", total_wl)

'''
Capacitance-based score:
score = total_wl + 2*[(buffer_input_cap)/wire_unit_cap]*buffer_count
'''

### Capacitance based score ###
score = total_wl + 2*header['buffer_input_cap']/header['unit_C']*num_bufs
print("Capacitance-based score =", round(score, 2))

end = time.time()

print(f"Execution time: {end - start:.2f} seconds")
