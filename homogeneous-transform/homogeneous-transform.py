import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    # Write code here
    T_arr = np.array(T)[:-1]
    x = np.array(points)

    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    out = np.einsum("ij,nj->ni", T_arr, x)
    return np.squeeze(out)
    