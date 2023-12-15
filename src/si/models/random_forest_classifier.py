import numpy as np
from typing import List, Tuple
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class RandomForestClassifier:
    '''
    Class representing a Random Forest Classifier model.
    '''

    def __init__(self, n_estimators: int, max_features: int = None, min_sample_split: int = 2, max_depth: int = 10, mode: str = 'gini', seed: int = 42) -> None:
        '''
        Creates a new instance of the RandomForestClassifier class.

        Parameters
        ----------
        n_estimators: int
            The number of trees in the forest.
        max_features: int
            The number of features to consider when looking for the best split.
        min_sample_split: int
            The minimum number of samples required to split an internal node.
        max_depth: int
            The maximum depth of the tree.
        mode: str
            The impurity measure to use.
        seed: int
            The seed to use for the random number generator.
        '''
        #parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        #estimated parameters
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        '''
        Fits the model to the given data.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        '''
        # seed the random number generator
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.shape()

        # set max_features to sqrt(n_features) if not set (we do this to reduce overfitting)
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # loop through the number of estimators (trees)
        for i in range(self.n_estimators):

            # create a bootstrap sample of the data
            bootstrap_samples_idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_features_idx = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_dataset = Dataset(dataset.X[bootstrap_samples_idx, :][:, bootstrap_features_idx], dataset.y[bootstrap_samples_idx])

            # fit a decision tree to the bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_sample_split=self.min_sample_split, mode=self.mode)
            tree.fit(bootstrap_dataset)

            # add the tree to the list of trees
            self.trees.append((bootstrap_features_idx, tree))
        return self
    
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        Predicts the target values for the given data.

        Parameters
        ----------
        dataset: Dataset
            The input data.

        Returns
        -------
        np.ndarray
            The predicted target values.
        '''
        # set up an array to store the predictions of each tree
        predictions = [None] * self.n_estimators

        # loop through the trees and get their predictions (only for the features they were trained on)
        for i, (features_idx, tree) in enumerate(self.trees):
            predictions[i] = tree.predict(Dataset(dataset.X[:, features_idx], dataset.y))
        
        # get the most frequent prediction for each sample
        most_frequent = []
        for z in zip(*predictions):
            most_frequent.append(max(set(z), key=z.count))

        return np.array(most_frequent)

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv('/home/pauloseixal/Github/si/datasets/iris/iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=5, min_sample_split=3, max_depth=3, mode='gini')
    model.fit(train)
    # print(model.predict(test))
    print(model.score(test))

    #sklearn random forest classifier
    from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
    
    model2 = SklearnRandomForestClassifier(n_estimators=5, min_samples_split=3, max_depth=3)
    model2.fit(train.X, train.y)
    # print(model2.predict(test.X))
    print(accuracy(test.y, model2.predict(test.X)))

