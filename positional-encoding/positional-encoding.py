import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    # positions (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]

    # dimension indices (1, d_model)
    dims = np.arange(d_model)[np.newaxis, :]

    # compute denominator term
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))

    # compute angles
    angles = pos * angle_rates

    # initialize encoding matrix
    pe = np.zeros((seq_len, d_model), dtype=float)

    # apply sin to even indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])

    # apply cos to odd indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe
    print(positional_encoding(3,4))
    