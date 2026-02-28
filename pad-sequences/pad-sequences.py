import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad or truncate sequences to equal length.

    Parameters:
    - seqs: list of lists of ints
    - pad_value: value used for padding
    - max_len: maximum length (auto-detected if None)

    Returns:
    - NumPy array of shape (N, max_len) with dtype=int
    """

    # If empty input
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)

    # Auto-detect max length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    N = len(seqs)

    # Initialize output array filled with pad_value
    padded = np.full((N, max_len), pad_value, dtype=int)

    # Fill with sequence values
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]          # truncate if too long
        padded[i, :len(trunc)] = trunc  # pad on right automatically

    return padded