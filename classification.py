#HW2_BayuFebriansyah
#KNN classification

import numpy as np


def fit_knn(X, y, n_neighbors):
    """Return the error rate and predicted labels on X, including self."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2 or X.size == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty matrix of finite numbers.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must contain one label for each row of X.")
    if isinstance(n_neighbors, (bool, np.bool_)) or not isinstance(n_neighbors, (int, np.integer)):
        raise TypeError("n_neighbors must be an integer.")
    if not 1 <= n_neighbors <= len(X):
        raise ValueError("n_neighbors must be between 1 and the number of rows.")

    predictions = np.empty(len(y), dtype=y.dtype)

    for row in range(len(X)):
        distances = np.sqrt(np.sum((X - X[row]) ** 2, axis=1))

        # Put self first, other equal distances keep their input order.
        distances[row] = -1
        neighbors = np.argsort(distances, kind="stable")[:n_neighbors]

        # A tied vote goes to the first label in sorted order.
        labels, counts = np.unique(y[neighbors], return_counts=True)
        predictions[row] = labels[np.argmax(counts)]

    error = float(np.mean(predictions != y))
    return error, predictions.tolist()
