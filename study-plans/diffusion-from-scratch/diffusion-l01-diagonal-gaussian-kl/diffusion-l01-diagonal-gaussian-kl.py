import numpy as np

def diagonal_gaussian_kl(mean_p: np.ndarray, log_var_p: np.ndarray, mean_q: np.ndarray, log_var_q: np.ndarray) -> np.ndarray:
    """
    Returns KL(p || q) summed over the last axis, as a float64 array with the leading shape of the inputs.
    """
    term1 = log_var_q - log_var_p
    term2 = (np.exp(log_var_p) + (mean_p - mean_q) ** 2) / np.exp(log_var_q)
    return np.sum(0.5 * (term1 + term2 - 1), axis=-1)