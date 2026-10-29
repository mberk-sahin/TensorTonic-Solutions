import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    for i in range(len(w)):
        s[i] = beta * np.array(s[i]) + (1-beta) * np.power(g[i], 2)
        w[i] -= lr / (np.sqrt(s[i] + eps)) * g[i]
    return w, s