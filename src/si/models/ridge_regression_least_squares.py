import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
# implement class

# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(alpha=2.0)
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))
