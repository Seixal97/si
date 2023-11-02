import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier:
    '''
    This class implements a stacking classifier to combine multiple models.
    '''

    def __init__(self, models: list, final_model: object) -> None:
        '''
        Creates a new instance of the StackingClassifier class.

        Parameters
        ----------
        models: list
            The list of models to use.
        final_model: object
            The final model to use.
        '''
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        '''
        Fits the model to the given data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        StackingClassifier
            The fitted model.
        '''
        for model in self.models:
            model.fit(dataset)
        
        predictions = list()

        for model in self.models:
            predictions.append(model.predict(dataset))
        
        predictions = np.array(predictions).T
        self.final_model.fit(Dataset(dataset.X, predictions))

        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        '''
        Predicts the labels for the given data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the labels for.

        Returns
        -------
        np.ndarray
            The predicted labels.
        '''
        predictions = list()

        for model in self.models:
            predictions.append(model.predict(dataset))
        
        predictions = np.array(predictions).T
        return self.final_model.predict(Dataset(dataset.X, predictions))
    
    def score(self, dataset: Dataset) -> float:
        '''
        Calculates the accuracy of the model on the given data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy for.

        Returns
        -------
        float
            The accuracy of the model.
        '''
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    data = read_csv('/home/pauloseixal/Github/si/datasets/breast_bin/breast-bin.csv', sep=",",features=True,label=True)
    train, test = stratified_train_test_split(data, test_size=0.20, random_state=42)

    #knnregressor
    knn = KNNClassifier(k=5)
    
    #logistic regression
    lr=LogisticRegression(l2_penalty=0.1, alpha=0.1, max_iter=1000)

    #decisiontreee
    dt=DecisionTreeClassifier(min_sample_split=2, max_depth=10, mode='gini')

    #final model
    final_model=KNNClassifier(k=5)
    modelos=[knn,lr,dt]
    exercise=StackingClassifier(modelos,final_model)
    exercise.fit(train)
    print(exercise.score(test))

    #sklearn
    from sklearn.ensemble import StackingClassifier as StackingClassifier_sklearn
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    #knnregressor
    knn = KNeighborsClassifier(n_neighbors=5)

    #logistic regression
    lr=LogisticRegression(penalty='l2', C=0.1, max_iter=1000)

    #decisiontreee
    dt=DecisionTreeClassifier(min_samples_split=2, max_depth=10, criterion='gini')

    #final model
    final_model=KNeighborsClassifier(n_neighbors=5)
    models=[('knn',knn),('lr',lr),('dt',dt)]
    exercise=StackingClassifier_sklearn(estimators=models,final_estimator=final_model)
    exercise.fit(train.X, train.y)
    print(accuracy(test.y, exercise.predict(test.X)))
