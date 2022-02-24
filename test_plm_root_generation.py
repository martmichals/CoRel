import os
from plm_root_generation import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

roots = []
with open('../../cmp/notebooks/corel_critical_analysis/data/potential_roots.txt', 'r') as f:
    for line in f:
        roots.append(line.strip())

ret_val = root_node_inference(
    ['seafood', 'dessert', 'burger', 'salad'],
    roots,
    './prompts.txt',
    max_spaces=1400
)

print(ret_val)

