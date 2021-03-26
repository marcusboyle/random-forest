# Standard library imports
from typing import Tuple

# Third party imports
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_data() -> Tuple[np.ndarray]:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test
