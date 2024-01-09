#!/bin/bash

# 遍历当前目录下的所有JSON文件
for file in *.json; do
    # 执行命令并将输出重定向到 out.txt
    python ../../simulator.py -f "$file" | tee -a out.txt
done
