# Standard library imports
from typing import Tuple

# Third party imports
import numpy as np
from scipy.sparse import data
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split


def load_data(dataset_name='boston') -> Tuple[np.ndarray]:
    if dataset_name == 'boston':
        # Load the 'Boston house prices' dataset for regression.
        # X: matrix of size (442, 10)
        # y: vector of size (442,)
        X_array, y_array = load_boston(return_X_y=True)
    elif dataset_name == 'diabetes':
        # Load the 'Diabetes' dataset for regression.
        # X: matrix of size (506, 13)
        # y: vector of size (506,)
        X_array, y_array = load_diabetes(return_X_y=True)
    else:
        raise ValueError('Invalid dataset_name')

    # Split the dataset into training and testing partitions.
    # 80% used for training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=0.2, shuffle=True
    )

    return X_train, X_test, y_train, y_test
