import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", type=int, help="start_gpu_id")
parser.add_argument("-g", type=int, help="num_group")
parser.add_argument("-a", type=int, help="avg_group_size")
parser.add_argument("-d", type=int, help="std")
args = parser.parse_args()

start_gpu_id = args.i
num_group = args.g
# 设定均值和标准差
avg_group_size = args.a  # 指定的均值
std = args.d   # 标准差可以自定义

for _ in range(num_group):
    

    # 生成一个满足正态分布的浮点数
    group_size = np.random.normal(avg_group_size, std)

    # 将浮点数转换为正整数
    group_size = int(max(5, np.round(group_size)))
    
    str = ", ".join(f"{i}" for i in range(start_gpu_id, start_gpu_id + group_size))
    start_gpu_id += group_size
    print(str)

