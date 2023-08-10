import numpy as np

def make_mask(x:int, y:int, a_list:np.ndarray) -> np.ndarray:
    mask = np.zeros((x, y))
    for i in range(x):
        mask[i, a_list == i+1] = 1
        
    return mask

def make_inc_matrix(start_node:np.ndarray, end_node:np.ndarray) -> np.ndarray:
    max_node = max(np.max(start_node), np.max(end_node))
    inc = np.zeros((max_node, len(start_node)))
    for j in range(len(start_node)):
        inc[start_node[j]-1, j] = 1
        inc[end_node[j]-1, j] = -1
        
    return inc