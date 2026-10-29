import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x_arr = np.array(x)
    sample_std = np.std(x, ddof=1)
    return {"variance": float(sample_std**2), "standard_deviation": float(sample_std)}