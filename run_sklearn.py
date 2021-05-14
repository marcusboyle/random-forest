# Standard library imports
import time
from typing import Dict, Union

# Third party imports
import yaml
from sklearn.ensemble import RandomForestRegressor

# Local imports
from random_forest.datasets import load_data
from random_forest.metrics import calc_Rsq


def run_sklearn(hyperparams: Dict[str, Union[int, float]]) -> None:
    X_train, X_test, y_train, y_test = load_data()

    start_time = time.time()
    print('\nRunning Regression Forest...')

    # Initialise instance of the RandomForest class
    rf_model = RandomForestRegressor(
        n_estimators=hyperparams['n_trees'],
        max_depth=hyperparams['max_depth'],
        max_samples=hyperparams['sample_ratio'],
        max_features='sqrt'
    )
    # Fit the model using training data
    rf_model.fit(X_train, y_train)

    # Predict on the training and testing data
    y_preds_train = rf_model.predict(X_train)
    y_preds_test = rf_model.predict(X_test)
    print(f'Done. Time taken: {time.time() - start_time:.2f}s\n')

    # Evaluate training and testing performance
    Rsq_train = calc_Rsq(y_train, y_preds_train)
    Rsq_test = calc_Rsq(y_test, y_preds_test)
    print(f'Train R2: {Rsq_train * 100:.2f}%')
    print(f'Test R2: {Rsq_test * 100:.2f}%')


if __name__ == '__main__':
    with open('hyperparams.yaml', 'r') as f:
        hyperparams = yaml.safe_load(f)

    run_sklearn(hyperparams)
