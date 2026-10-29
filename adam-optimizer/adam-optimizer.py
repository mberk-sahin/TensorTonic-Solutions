import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    for i in range(len(param)):
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
        bias_correct = m[i] / (1 - beta1 ** t)
        v_correct = v[i] / (1 - beta2 ** t)
        param[i] -= lr * bias_correct / (np.sqrt(v_correct) + eps)
    return (param, m, v)