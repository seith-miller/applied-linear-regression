import numpy as np

def compute_cost(X, y, theta):
    """
    Compute the cost (mean squared error) for linear regression.

    Parameters:
    X : numpy.ndarray
        The input feature matrix (with intercept term added).
    y : numpy.ndarray
        The target variable.
    theta : numpy.ndarray
        The parameters of the linear regression model (intercept and slope).

    Returns:
    float
        The computed cost (mean squared error).
    """
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost
