import numpy as np

def gaussian_mixture_score(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """
    Returns the score grad_x log p(x) of the isotropic Gaussian mixture at every row of x, as a float64 array of shape (N, D).
    """

    N, D = x.shape

    # adjust the dimensions for algebra
    x = np.expand_dims(x, axis=1)
    weights = np.expand_dims(weights, axis=(0,2))
    means = np.expand_dims(means, axis=0)
    variances = np.expand_dims(variances, axis=(0,2))

    # calculate responsibilities
    mean_diff = x - means
    log_gauss = -0.5 * D * np.log(2 * np.pi * variances) - np.sum(mean_diff ** 2, axis=-1, keepdims=True) / (2 * variances)

    energy = np.log(weights) + log_gauss
    energy = np.exp(energy - np.max(energy, axis=1, keepdims=True))
    resp = energy / np.sum(energy, axis=1, keepdims=True)

    # calculate score
    score = np.sum(-resp * mean_diff / variances, axis=1)
    return score
    
        
    
