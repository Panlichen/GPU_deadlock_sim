import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", type=int, help="avg_coll_cnt")
parser.add_argument("-d", type=int, help="std")
parser.add_argument("-g", type=int, help="num_group")

args = parser.parse_args()

num_group = args.g
avg_coll_cnt = args.a  # 指定的均值
std = args.d   # 标准差可以自定义

for _ in range(num_group):
    # 生成一个满足正态分布的浮点数
    coll_cnt = np.random.normal(avg_coll_cnt, std)

    # 将浮点数转换为正整数
    coll_cnt = int(max(120, np.round(coll_cnt)))
    print(f"{coll_cnt},")

