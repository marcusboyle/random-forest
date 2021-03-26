# Standard library imports
import time

# Third party imports
import yaml
import numpy as np

# Local imports
from random_forest.datasets import load_data
from random_forest.regression import RandomForestRegression
from random_forest.metrics import calc_Rsq


with open('hyperparams.yaml', 'r') as f:
    hyperparams = yaml.safe_load(f)

def max_features(n_features): return (n_features - 1) // 3

X_train, X_test, y_train, y_test = load_data()

# Run algorithm
start_time = time.time()
print('Running Regression Forest...')
rf_model = RandomForestRegression(
    **hyperparams,
    max_features=max_features
)
rf_model.fit(X_train, y_train)

predictions_train = rf_model.predict(X_train)
predictions_test = rf_model.predict(X_test)
print(f'Done. Time taken: {time.time() - start_time:.2f}s\n')

# Evaluate train and test performance
Rsq_train = calc_Rsq(y_train, predictions_train)
Rsq_test = calc_Rsq(y_test, predictions_test)
print(f'Train accuracy: {Rsq_train*100:.2f}%')
print(f'Test accuracy: {Rsq_test*100:.2f}%')

###
from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor(
    n_estimators=hyperparams['n_trees'],
    max_depth=hyperparams['max_depth'],
    max_samples=hyperparams['sample_ratio'],
    max_features='sqrt'
)
rf_model.fit(X_train, y_train)

start_time = time.time()
print('\nRunning Regression Forest...')
predictions_train = rf_model.predict(X_train)
predictions_test = rf_model.predict(X_test)
print(f'Done. Time taken: {time.time() - start_time:.2f}s\n')

# Evaluate train and test performance
Rsq_train = calc_Rsq(y_train, predictions_train)
Rsq_test = calc_Rsq(y_test, predictions_test)
print(f'Train accuracy: {Rsq_train*100:.2f}%')
print(f'Test accuracy: {Rsq_test*100:.2f}%')
