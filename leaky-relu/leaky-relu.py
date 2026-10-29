import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    # Write code here
    x_arr = np.array(x)
    return np.where(x_arr >= 0, x_arr, alpha * x_arr)