from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import numpy as np


class CategoricalNB:
    '''
    Model that implements the Naive Bayes algorithm for categorical data. This class will be used only for binary features
    '''

    def __init__(self, smoothing: float = 1.0):
        '''
        Initialize the model

        Parameters
        ----------
        smoothing: float
            Smoothing parameter to avoid zero probabilities

        Attributes
        ----------
        class_prior: list
            Prior probabilities for each class
        feature_prob: list
            Conditional probabilities for each feature given the class
        '''
        # Parameters
        self.smoothing = smoothing

        # Attributes
        self.class_prior = list()
        self.feature_prob = list()

    def fit(self, dataset: Dataset) -> 'CategoricalNB':
        '''
        Fit the model using the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model with
        
        Returns
        -------
        self: CategoricalNB
        '''
        # define n_samples, n_features and n_classes
        n_samples, n_features = dataset.X.shape
        n_classes = len(dataset.get_classes())

        # initialize class_count (number of samples for each class), feature_count (sum of each feature for each class)
        # and class_prior (prior probability for each class)
        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)


        # compute class_count, feature_count and class_prior
        for i, sample in enumerate(dataset.X):
            class_count[dataset.y[i]] += 1
            feature_count[dataset.y[i]] += sample

        self.class_prior = class_count / n_samples

        # add smoothing to avoid zero probabilities
        class_count += self.smoothing
        feature_count += self.smoothing

        # compute feature_prob (feature_counts divided by class_counts for each class)
        self.feature_prob = feature_count / class_count.reshape(-1,1)

        return self
    

    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        Predict the labels of the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the labels of

        Returns
        ----------
        y_pred: np.ndarray
            The predicted labels
        '''
        # define class_prob (probability of each class) and y_pred (predicted labels)
        n_classes = len(dataset.get_classes())
        class_prob = np.zeros(n_classes)
        y_pred = np.zeros(len(dataset.X))
        
        
        # for each sample in the dataset compute the probability of each class
        for i, sample in enumerate(dataset.X):
            for j in range(n_classes):
                class_prob[j] = np.prod(sample * self.feature_prob[j] + (1 - sample) * (1 - self.feature_prob[j])) * self.class_prior[j]
            y_pred[i] = np.argmax(class_prob)

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
        y_pred = self.predict(dataset)

        return accuracy(dataset.y, y_pred)




if __name__ == '__main__':
    # Create a dataset and apply the model
    from si.model_selection.split import train_test_split
    dataset = Dataset.from_random(100, 20, 2)
    dataset_train, dataset_test = train_test_split(dataset, 0.2)

    model = CategoricalNB()
    model.fit(dataset_train)
    print('Model accuracy:', model.score(dataset_test))

    # Compare with sklearn
    from sklearn.naive_bayes import CategoricalNB as CategoricalNB_sk
    model_sk = CategoricalNB_sk()
    model_sk.fit(dataset_train.X, dataset_train.y)
    print('Model accuracy (sklearn):', model_sk.score(dataset_test.X, dataset_test.y))


    
