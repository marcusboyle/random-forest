# Standard library imports
import random
from typing import Union, List, Dict, Any, Callable

# Third party imports
import numpy as np


class RandomForestRegression:
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 4,
        sample_ratio: float = 0.6,
        max_features: Callable[[int], int] = lambda n_features: (n_features - 1) // 3
    ) -> None:

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.max_features = max_features
        self.trees = None

    def fit(self, X_array: np.ndarray, y_array: np.ndarray) -> None:
        """
        Train the Random Forest.
        """
        assert len(X_array) == len(y_array)

        self.trees = []
        print(f'Building trees...')
        n_rows = len(y_array)
        for _ in range(self.n_trees):
            # First, select the sample used for this tree
            sample_indices = self.__subsample(n_rows)
            X_sample = X_array[sample_indices]
            y_sample = y_array[sample_indices]

            # Build out the tree and store it
            tree = self.__build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)
        print(f'Trees built.')

    def predict(self, X_array: np.ndarray) -> List[float]:
        if self.trees is None:
            print('You must fit the model first.')
            return []
        else:
            predictions = [
                self.__bagging_predict(self.trees, row) 
                for row in X_array
            ]
            return predictions

    def __find_split(self, X_array: np.ndarray, y_array: np.ndarray) -> Dict[str, Any]:
        best = {'cost': np.inf}
        n_features = X_array.shape[1]
        n_features_sample = self.max_features(n_features)
        # Randomly sample features with replacement
        feature_indices = random.sample(range(n_features), n_features_sample)

        # Loop every possible split of every dimension...
        for i in feature_indices:  # one for each feature randomly chosen above
            for split in np.unique(X_array[:, i]):  # For each unique value in the given column, test a split at that value
                left_indices = np.where(X_array[:, i] <= split)[0]
                right_indices = np.where(X_array[:, i] > split)[0]

                # Compute the cost by combining the TSS for each side of the split
                cost = 0
                for indices in [left_indices, right_indices]:
                    if len(indices) != 0:
                        # TSS is used to measure the variance of each split
                        TSS = np.sum((y_array[indices] - np.mean(y_array[indices]))**2)
                        cost += TSS

                # Update values if the cost is less than best cost
                if cost < best['cost']:
                    best = {
                        'feature': i,
                        'split': split,
                        'cost': cost,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        return best

    def __build_tree(self, X_array: np.ndarray, y_array: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        # Stopping conditions: max depth reached, or all values remaining are the same
        # If so generate a leaf node...
        if depth == self.max_depth or (y_array == y_array[0]).all():
            # Generate a leaf node...
            # classes, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'value': np.mean(y_array)}

        else:
            # Find a good split for this node
            move = self.__find_split(X_array, y_array)
            left_indices = move['left_indices']
            right_indices = move['right_indices']

            # If all values will be put in the same child node, then create leaf node
            if len(right_indices) == 0:
                return {'left': True, 'value': np.mean(y_array)}
            # Else, we'll continue to build out the tree further
            left = self.__build_tree(X_array[left_indices], y_array[left_indices], depth + 1)
            right = self.__build_tree(X_array[right_indices], y_array[right_indices], depth + 1)

            node = {
                'leaf': False,
                'feature': move['feature'],
                'split': move['split'],
                'cost': move['cost'],
                'left': left,
                'right': right
            }
            return node

    def __subsample(self, n_rows: int) -> List[int]:
        # Randomly sample with replacement, according to the sample ratio
        n_sample = round(n_rows * self.sample_ratio)
        sample_indices = random.choices(range(n_rows), k=n_sample)
        return sample_indices

    def __bagging_predict(self, trees: List[dict], x_row: np.ndarray) -> float:
        # Given a row of x values, make a prediction for y with each tree.
        # The predictions will be averaged to give our overall prediction.
        predictions = [self.__predict_one(tree, x_row) for tree in trees]
        prediction = np.mean(predictions)
        return prediction

    def __predict_one(self, tree: dict, x_row: np.ndarray) -> float:
        """Does the prediction for a single data point."""
        # If we're at a leaf node, return the value. If not, then move either
        # left or right down the tree, depending on the value of the relevant
        # feature
        if tree['leaf']:
            return tree['value']
        else:
            if x_row[tree['feature']] <= tree['split']:
                return self.__predict_one(tree['left'], x_row)
            else:
                return self.__predict_one(tree['right'], x_row)
