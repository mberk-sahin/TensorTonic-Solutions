import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    exp = np.exp(-2 * np.array(x))
    return (1 - exp) / (1 + exp)