from typing import Tuple, Union

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray],
                                                Tuple[float, float]]:
    """
    Scoring function for classification problems. It computes one-way ANOVA F-value for the
    provided dataset. The F-value scores allows analyzing if the mean between two or more groups (factors)
    are significantly different. Samples are grouped by the labels of the dataset.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F scores
    p: np.array, shape (n_features,)
        p-values
    """
    classes = dataset.get_classes()

    # group samples by each unique class
    groups = [dataset.X[dataset.y == c] for c in classes]

    # compute F-value and p-value between each group
    F, p = stats.f_oneway(*groups)
    return F, p


if __name__ == '__main__':
    data = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]))
    print(f_classification(data))
