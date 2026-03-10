import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    
    # ensure numpy arrays
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Step 1: update running average of squared gradients
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: parameter update
    new_w = w - lr * g / (np.sqrt(new_s) + eps)

    return new_w, new_s