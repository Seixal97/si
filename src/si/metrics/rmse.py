import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Computes the root mean squared error between the true and predicted values

    Parameters
    ----------
    y_true: np.ndarray
        The real values of y
    y_pred: np.ndarray
        The predicted values of y

    Returns
    -------
    rmse: float
        Float corresponding to the error between the real and predicted values of y
    '''
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


if __name__ == '__main__':
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    print(rmse(y_true, y_pred))