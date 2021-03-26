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

    def find_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        best = {'cost': np.inf}
        n_features = X.shape[1]
        n_features_sample = self.max_features(n_features)
        # Randomly sample features with replacement
        feature_indices = random.sample(range(n_features), n_features_sample)

        # Loop every possible split of every dimension...
        for i in feature_indices:  # one for each feature randomly chosen above
            for split in np.unique(X[:, i]):  # For each unique value in the given column, test a split at that value
                left_indices = np.where(X[:, i] <= split)[0]
                right_indices = np.where(X[:, i] > split)[0]

                # Compute the cost by combining the TSS for each side of the split
                cost = 0
                for indices in [left_indices, right_indices]:
                    if len(indices) != 0:
                        # TSS is used to measure the variance of each split
                        TSS = np.sum((y[indices] - np.mean(y[indices]))**2)
                        cost += TSS

                # left_SSE = np.sum(
                #     (y[left_indices] - np.mean(y[left_indices]))**2
                # )
                # # Avoid finding mean of zero
                # if len(right_indices) != 0:
                #     right_SSE = np.sum(
                #         (y[right_indices] - np.mean(y[right_indices]))**2
                #     )
                # else:
                #     right_SSE = 0

                # # cost = left_SSE/len(left_indices) + right_SSE/len(right_indices)
                # cost = left_SSE + right_SSE

                # Update values if the cost is less than best cost
                if cost < best['cost']:
                    best = {
                        'feature': i,
                        'split': split,
                        'cost': cost,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }

        # left_size = len(best['left_indices'])
        # right_size = len(best['right_indices'])
        # if left_size == 0:
        #     print(f'Left empty. Cost: {best["cost"]}')
        #     print(best['left_indices'])
        # if right_size == 0:
        #     print(f'Right empty. Cost: {best["cost"]}')
        #     print(best['right_indices'])

        return best

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        # Stopping conditions: max depth reached, or all values remaining are the same
        # If so generate a leaf node...
        if depth == self.max_depth or (y == y[0]).all():
            # Generate a leaf node...
            # classes, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'value': np.mean(y)}
        
        else:
            # Find a good split for this node
            move = self.find_split(X, y)
            left_indices = move['left_indices']
            right_indices = move['right_indices']

            # If all values will be put in the same child node, then create leaf node
            if len(right_indices) == 0:
                return {'left': True, 'value': np.mean(y)}
            # Else, we'll continue to build out the tree further
            left = self.build_tree(X[left_indices], y[left_indices], depth + 1)
            right = self.build_tree(X[right_indices], y[right_indices], depth + 1)
            
            node = {
                'leaf': False,
                'feature': move['feature'],
                'split': move['split'],
                'cost': move['cost'],
                'left': left,
                'right': right
            }
            return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest.
        """
        assert len(X) == len(y)

        self.trees = []
        print(f'Building trees...')
        for _ in range(self.n_trees):
            n_rows = len(y)
            # First, select the sample used for this tree
            sample_indices = self.subsample(n_rows)
            X_sample = X[sample_indices]  # .reshape(-1,1)
            y_sample = y[sample_indices]

            # Build out the tree and store it
            tree = self.build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)
        print(f'Trees built.')

    def predict(self, X: np.ndarray) -> List[float]:
        if self.trees is None:
            print('You must fit the model first.')
            return []
        else:
            predictions = [
                self.bagging_predict(self.trees, row) 
                for row in X
            ]
            return predictions

    def subsample(self, n_rows: int) -> List[int]:
        # Randomly sample with replacement, according to the sample ratio
        n_sample = round(n_rows * self.sample_ratio)
        sample_indices = random.choices(range(n_rows), k=n_sample)
        return sample_indices

    def bagging_predict(self, trees: List[dict], x: np.ndarray) -> float:
        # Given a row of x values, make a prediction for y with each tree.
        # The predictions will be averaged to give our overall prediction.
        predictions = [self.predict_one(tree, x) for tree in trees]
        prediction = np.mean(predictions)
        return prediction

    def predict_one(self, tree: dict, x: np.ndarray) -> float:
        """Does the prediction for a single data point"""
        # If we're at a leaf node, return the value. If not, then move either
        # left or right down the tree, depending on the value of the relevant
        # feature
        if tree['leaf']:
            return tree['value']
        else:
            if x[tree['feature']] <= tree['split']:
                return self.predict_one(tree['left'], x)
            else:
                return self.predict_one(tree['right'], x)
