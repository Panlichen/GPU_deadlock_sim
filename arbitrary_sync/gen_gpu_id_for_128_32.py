import numpy as np
import argparse
start_gpu_id = 64
num_group_size3 = 32
# num_group_size8 = 4

for _ in range(num_group_size3):

    group_size = 2
    
    str = ", ".join(f"{i}" for i in range(start_gpu_id, start_gpu_id + group_size))
    start_gpu_id += group_size
    print(str)

# for _ in range(num_group_size8):

#     group_size = 8
    
#     str = ", ".join(f"{i}" for i in range(start_gpu_id, start_gpu_id + group_size))
#     start_gpu_id += group_size
#     print(str)
