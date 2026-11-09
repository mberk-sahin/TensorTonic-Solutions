import numpy as np

def cosine_noise_schedule(num_steps: int, s: float, max_beta: float) -> dict:
    """
    Returns a dict with float64 arrays "betas" and "alpha_bars", each of shape (num_steps,).
    """

    f = lambda u: np.cos((u/num_steps + s) / (1 + s) * np.pi / 2) ** 2

    t = np.arange(num_steps)
    betas = np.minimum(1 - f(t+1)/f(t), max_beta)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    return {
        "betas": betas,
        "alpha_bars": alpha_bars
    }
    
    
