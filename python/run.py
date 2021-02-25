import time
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from src.rf_regression import RandomForestRegression
from src.evaluation import calc_Rsq


# X, y = load_boston(return_X_y=True)
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Hyperparameters
n_trees = 10
max_depth = 4
sample_ratio = 0.6
def max_features(n_features): return (n_features - 1) // 3


# Run algorithm
start_time = time.time()
print('Running Regression Forest...')
rf_model = RandomForestRegression(
    n_trees,
    max_depth,
    sample_ratio,
    max_features
)
rf_model.fit(X, y)

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
    n_estimators=n_trees,
    max_depth=max_depth,
    max_samples=sample_ratio,
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
