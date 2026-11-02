import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    n_samples = len(X_test)

    counts = {}
    max_label, max_count = -1, -1
    labels = sorted(list(set(y_train)))
    for i in range(len(y_train)):
        label = y_train[i]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

        if counts[label] > max_count:
            max_label = label
            max_count = counts[label]
    
    return np.ones(n_samples, dtype=np.int32) * max_label

    