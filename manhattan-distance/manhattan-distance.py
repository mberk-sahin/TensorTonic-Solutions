import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    x_arr = np.array(x, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    
    return float(np.abs(x_arr - y_arr).sum().item())