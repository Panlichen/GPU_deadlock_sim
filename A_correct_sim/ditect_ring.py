# 这段代码是一个用于检测图中环的Python脚本，采用深度优先搜索（DFS）算法。其基本原理和执行过程如下：

# 图的表示：该脚本使用字典graph来表示图。每个键是一个顶点，与之关联的值是一个列表，包含与该顶点相邻的顶点。例如，"a": ["b", "c"]意味着顶点a与顶点b和c相连。

# DFS算法：深度优先搜索（DFS）是一种用于遍历或搜索树或图的算法。该算法从一个顶点开始，探索尽可能深的分支，直到找不到未访问的相邻顶点为止，然后回溯。

# 检测环：在DFS过程中，如果访问到一个已经访问过的顶点，且该顶点在当前的访问路径上（即在trace列表中），那么就检测到一个环。trace列表维护着当前路径上的顶点。

# 实现细节：

# vis列表用于存储访问过的顶点。
# trace列表用于跟踪当前DFS路径上的顶点。
# dfs函数用于执行深度优先搜索。它接受一个顶点v作为参数。
# 如果顶点v已在vis中，表示已访问过：
# 如果v也在trace中，表示找到一个环，打印出环的路径。
# 否则，返回，因为不需要再探索从此顶点出发的路径。
# 将v加入vis和trace。
# 对于顶点v的每一个相邻顶点，递归调用dfs函数。
# 完成对所有相邻顶点的探索后，从trace中移除顶点v。

PRINT_RING = True

vis = []
trace = []
find_ring = False

def dfs(graph, v):
    global find_ring
    global vis
    global trace
    if v in vis:
        if v in trace:
            find_ring = True
            if PRINT_RING:
                v_index = trace.index(v)
                print("有环：")
                for i in range(v_index, len(trace)):
                    print(trace[i] + ' ', end='')
                print(v)
                print("\n")
            return
        return

    vis.append(v)
    trace.append(v)
    for vs in graph[v]:
        dfs(graph, vs)
        if find_ring:
            return
    trace.pop()


def detect_ring(graph) -> bool:
    for v in graph.keys():
        dfs(graph, v)
        if find_ring:
            return True
    return False