import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    '''
    The Ridge Regression Least Squares model is a linear model that uses the Least Squares method to fit the data.

    This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm.
    
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not
        
    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    '''

    def __init__(self, l2_penalty: float = 1, scale: bool = True):
        '''
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not               
        
        '''
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        '''
        Fit the model using the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model with

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        '''
        # scale the dataset
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # add intercept term to X (add column of ones in the beginning)
        X = np.c_[np.ones(X.shape[0]), X]

        # compute penalty matrix (l2_penalty * identity matrix)
        penalty_matrix = np.eye(X.shape[1]) * self.l2_penalty

        # change first position of penalty matrix to 0 (we don't want to regularize the intercept term theta_zero)
        penalty_matrix[0, 0] = 0

        # compute theta_zero (first element of the theta vector) and theta (remaium elements)
        theta_vector = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(dataset.y)
        self.theta_zero = theta_vector[0]
        self.theta = theta_vector[1:]

        return self
    
    def predict(self, dataset: Dataset) -> np.array:
        '''
        Predict the labels of the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the labels of

        Returns
        -------
        y_pred: np.array
            The predicted labels
        '''
        # scale the dataset
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # add intercept term to X (add column of ones in the beginning)
        X = np.c_[np.ones(X.shape[0]), X]

        # compute the predictions
        y_pred = X.dot(np.r_[self.theta_zero, self.theta])

        return y_pred
    
    def score(self, dataset: Dataset) -> float:
        '''
        Compute the score of the model on the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on

        Returns
        -------
        score: float
            The score of the model on the given dataset
        '''
        return mse(dataset.y, self.predict(dataset))


# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares()
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge()
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))
