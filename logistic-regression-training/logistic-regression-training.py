import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here

    # initialize the parameter 
    w, b = np.zeros((X.shape[-1],)), np.zeros((1,))

    for i in range(steps):

        # forward pass
        p = _sigmoid(np.matmul(X, w) + b)

        # backward pass 

        # update w
        w -= lr * np.matmul(X.T, (p - y)) / len(X)
        b -= lr * np.mean(p - y)

    return w, b.item()