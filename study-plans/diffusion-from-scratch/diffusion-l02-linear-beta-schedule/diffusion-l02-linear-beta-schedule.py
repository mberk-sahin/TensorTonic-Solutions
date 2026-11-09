import numpy as np

def linear_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> dict:
    """
    Returns a dict with float64 arrays "betas", "alphas" and "alpha_bars", each of shape (num_steps,).
    """
    t = np.arange(num_steps)
    if num_steps == 1:
        betas = np.array([beta_start])
    else:
        betas = beta_start + (t / (num_steps - 1)) * (beta_end - beta_start)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars
    }
