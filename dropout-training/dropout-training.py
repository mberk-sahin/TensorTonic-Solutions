import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.array(x)
    # Write code here
    samples = rng.random(x.shape)
    
    pattern = np.where(samples < p, 0, 1 / (1-p))
    out = x * pattern
    return out, pattern
    
    