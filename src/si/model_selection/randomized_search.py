import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: dict, scoring: callable = accuracy, cv: int = 5, n_ite: int = 10) -> dict:
    '''
    Implements a parameter optimization strategy with cross validation
    using a random number of combinations from the hyperparameter grid.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: dict
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds.
    n_ite: int
        The number of iterations to perform.

    Returns
    -------
    best_params: dict
        Dictionary with the results of the cross validation. Includes the scores, best parameters and the best model.
    '''

    # check if the provided hyperparameter grid is valid
    if not isinstance(hyperparameter_grid, dict):
        raise TypeError('The hyperparameter grid must be a dictionary.')
    
    for parameter in hyperparameter_grid.keys():
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
        
    randomized_search_output = {'hyperparameters': [], 
                                'scores': [], 
                                'best_hyperparameters': None, 
                                'best_scores': 0}
        
    # get n_iter hyperparameter combinations from the grid
    for i in range(n_ite):
        random_params = {}
        # get a random combination of hyperparameters from the given grid
        for key in hyperparameter_grid.keys():
            random_params[key] = np.random.choice(hyperparameter_grid[key])

        # set the model's hyperparameters based on the random combination selected above
        for key in random_params.keys():
            setattr(model, key, random_params[key])

        # perform cross validation
        model_cv_scores = k_fold_cross_validation(model, dataset, scoring, cv)

        # save the results
        randomized_search_output['hyperparameters'].append(random_params)
        randomized_search_output['scores'].append(model_cv_scores)

        # get the average score
        avg_score = np.mean(model_cv_scores)

        # check if the current model is the best one
        if avg_score > randomized_search_output['best_scores']:
            randomized_search_output['best_scores'] = avg_score
            randomized_search_output['best_hyperparameters'] = random_params
        
    return randomized_search_output


if __name__ == '__main__':
    from si.models.logistic_regression import LogisticRegression
    from si.model_selection.grid_search import grid_search_cv
    from si.io.csv_file import read_csv
    

    # load the dataset
    dataset = read_csv('/home/pauloseixal/Github/si/datasets/breast_bin/breast-bin.csv', sep=",",features=True,label=True)

    # define the model
    model = LogisticRegression()

    # define the hyperparameter grid
    hyperparameter_grid = {'l2_penalty': np.linspace(1, 10, 10),
                           'alpha': np.linspace(0.001, 0.0001, 100),
                           'max_iter': np.linspace(1000, 2000, 200),
                           }
    # print(hyperparameter_grid)

    # perform grid search cross validation
    results = randomized_search_cv(model=model, dataset=dataset, hyperparameter_grid=hyperparameter_grid, cv=3, n_ite=10)

    # print the results
    print('Grid search results:\n')

    print(f'Best score:\n {results["best_scores"]}')
    print()
    print(f'Best hyperparameters:\n {results["best_hyperparameters"]}')
    print()
    print(f'All scores:\n {results["scores"]}')
    print()
    print(f'All hyperparameters:\n {results["hyperparameters"]}')