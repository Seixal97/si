from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor:
    '''
    KNN Regressor
    the k-nearest neighbors regressor is a supervised learning algorithm that predicts the value of a new sample
    based on the k-nearest samples in the training data.
    This KNNRegressor is similar to the KNNClassifier, but is more suitable for regression problems.
    Therefore, it estimates the average value of the k most similar examples instead of the most common class.
    
    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use
    
    Attributes
    ----------
    dataset: np.ndarray
        The training data
    '''

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        '''
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        '''
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        '''
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        '''
        self.dataset = dataset
        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        '''
        It returns the label of the most similar sample in the dataset

        Parameters
        ----------
        sample: np.ndarray
            The sample to predict

        Returns
        -------
        label: Union[int, str]
            The label of the most similar sample in the dataset
        '''
        # get the distances between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)
        
        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # get the average label
        label = np.mean(k_nearest_neighbors_labels)
        return label
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        It predicts the labels of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict

        Returns
        -------
        y_pred: np.ndarray
            The predicted labels
        '''
        y_pred = np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        return y_pred
    
    def score(self, dataset: Dataset) -> float:
        '''
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to score

        Returns
        -------
        accuracy: float
            The RMSE of the model
        '''
        y_pred = self.predict(dataset)
        return rmse(dataset.y, y_pred)