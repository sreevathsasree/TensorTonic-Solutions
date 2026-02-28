import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    # Convert to arrays and get shapes
    X = np.array(X)
    y = np.array(y)
    N, D = X.shape

    # 1️⃣ Initialize parameters
    w = np.zeros(D)
    b = 0.0

    # 2️⃣ Gradient Descent Loop
    for _ in range(steps):
        # Linear combination
        z = X @ w + b     # shape: (N,)

        # Predictions
        p = _sigmoid(z)   # shape: (N,)

        # Compute gradients
        dw = (1/N) * (X.T @ (p - y))   # shape: (D,)
        db = (1/N) * np.sum(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b
