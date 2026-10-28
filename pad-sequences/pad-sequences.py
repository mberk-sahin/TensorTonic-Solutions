import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if len(seqs) == 0:
        return np.empty((0,0), dtype=np.int32)
    
    # Your code here
    if max_len is None:
        max_len = max(list(map(lambda x: len(x), seqs)))  # fimd maximum length

    N = len(seqs)
    tokens = np.ones((N, max_len), dtype=np.int32) * pad_value

    for i in range(N):
        seq_list, seq_len = seqs[i], len(seqs[i])
        if seq_len >= max_len:
            tokens[i] = np.array(seq_list[:max_len])
        elif seq_len < max_len:
            tokens[i,:seq_len] = np.array(seq_list)
    
    return tokens