import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    # Write code here
    X_arr = np.array(X)
    mu = np.mean(X_arr, axis=axis, keepdims=True)
    std = np.std(X_arr, axis=axis, keepdims=True)
    den = np.where(std < eps, eps, std)
    z = (X_arr - mu) / std
    return np.where(std < eps, 0, z)
    