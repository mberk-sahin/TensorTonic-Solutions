import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    if d_model == 0 or seq_len == 0:
        return np.empty((seq_len, d_model))

    pos_idxs = np.expand_dims(np.arange(seq_len), axis=1)

    d_model_ceil = d_model
    if d_model % 2 == 1:
        d_model_ceil += 1
    
    angle = pos_idxs / np.power(base, 2 * np.arange(d_model_ceil // 2) / d_model)
    pos_embs = np.stack((np.sin(angle), np.cos(angle)), axis=-1).reshape(seq_len, -1)

    if d_model_ceil != d_model:
        pos_embs = pos_embs[:,:-1]
    return pos_embs