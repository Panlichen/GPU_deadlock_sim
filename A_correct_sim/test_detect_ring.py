from ditect_ring import detect_ring

graph = {
    "a": ["b", "c"],
    "b": ["d"],
    "c": ["a", "d"],
    # "d": [ "e"],
    "d": ["c", "e"],  # add loop
    "e": ["d"]
}


# graph = { # 不连通
#     "a": ["b", "c"],
#     "b": ["c"],
#     "c": [],
#     # "d": [ "e"],
#     "d": ["e"],  # add loop
#     "e": ["f"],
#     "f": ["d"]
# }

print(detect_ring(graph))