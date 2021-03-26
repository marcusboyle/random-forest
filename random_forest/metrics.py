def calc_Rsq(y_train, y_pred):
    # Residual Sum of Squares (i.e. sum of squared errors)
    RSS = ((y_train - y_pred)**2).sum()
    # Total Sum of Squares (proportional to data's variance)
    TSS = ((y_train - y_train.mean())**2).sum()
    # R squared (1 minus unexplained variance)
    Rsq = 1 - (RSS/TSS)
    return Rsq
