from typing import Callable, Union, Literal

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.

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
    """
    def __init__(self, k: int = 1, weights: Literal['uniform', 'distance'] = 'uniform', distance: Callable = euclidean_distance):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        weights: Literal['uniform', 'distance']
            The weight function to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.weights = weights
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNClassifier
            The fitted model
        """
        self.dataset = dataset
        return self
    
    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        '''
        It returns the weights of the k nearest neighbors

        Parameters
        ----------
        distances: np.ndarray
            The distances between the sample and the dataset

        Returns
        -------
        weights: np.ndarray
            The weights of the k nearest neighbors
        '''
        # get the k nearest neighbors (first k indexes of the sorted distances)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        distances[k_nearest_neighbors] = np.maximum(distances[k_nearest_neighbors], 0.000001)

        # get the weights of the k nearest neighbors
        weights = 1 / distances[k_nearest_neighbors]
        return weights
    
    def _get_weighted_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the weighted label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the weighted label of

        Returns
        -------
        label: str or int
            The weighted label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the weights of the k nearest neighbors
        weights = self._get_weights(distances)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # get the weighted label
        if self.weights == 'uniform':
            return np.bincount(k_nearest_neighbors_labels).argmax()
        elif self.weights == 'distance':
            return np.bincount(k_nearest_neighbors_labels, weights=weights).argmax()

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        if self.weights == 'distance':
            return self._get_weighted_label(sample)
        
        else:
            # compute the distance between the sample and the dataset
            distances = self.distance(sample, self.dataset.X)

            # get the k nearest neighbors
            k_nearest_neighbors = np.argsort(distances)[:self.k]

            # get the labels of the k nearest neighbors
            k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

            # get the most common label
            labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
            return labels[np.argmax(counts)]
        

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        # dataset.X -> new samples to be tested against the dataset previously fitted to the model (self.dataset)
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3, weights='distance')

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')
  

