
from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    """
    Select features according to the specified percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: int, default=20
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: int = 20):
        """
        Select features according to the percentile.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int, default=20
            Percentile of top features to select.
        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        # compute F scores and p-values between each label for each feature
        self.F, self.p = self.score_func(dataset)
        self.F = np.nan_to_num(self.F)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the features in a specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the highest scoring features within a specified percentile.
        """

        # compute threshold value based on the specified percentile
        threshold = np.percentile(self.F, 100 - self.percentile)

        # find the indexes of the features with a score higher than the threshold value
        idxs = np.where(self.F > threshold)[0]

        # select the features based on the indexes
        features = np.array(dataset.features)[idxs]

        # create and return a new dataset with the selected features
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting the highest scoring features within a specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the highest scoring features within a specified percentile.
        """
        self.fit(dataset)
        return self.transform(dataset)
    

if __name__ == '__main__':
    from si.data.dataset import Dataset
    
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    print(len(dataset.features))
    selector = SelectPercentile(percentile=20)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)