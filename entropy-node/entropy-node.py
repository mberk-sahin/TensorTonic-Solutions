import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    _, unique_counts = np.unique(y, return_counts=True)
    probs = unique_counts / len(y)
    log2_probs = np.where(probs <= 0, 1, np.log2(probs))
    return float(-np.dot(probs, log2_probs).item())
    