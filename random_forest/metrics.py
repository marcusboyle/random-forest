import numpy as np


def calc_Rsq(y_truth: np.ndarray, y_pred: np.ndarray) -> float:
    # Residual Sum of Squares (i.e. sum of squared errors)
    RSS = ((y_truth - y_pred)**2).sum()
    # Total Sum of Squares (proportional to data's variance)
    TSS = ((y_truth - y_truth.mean())**2).sum()
    # R squared (1 minus unexplained variance)
    Rsq = 1 - (RSS/TSS)
    return Rsq
