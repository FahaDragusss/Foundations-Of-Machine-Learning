import numpy as np

def mean_absolute_error(y_pred , y_test_or_train):
    m = len(y_pred)
    sum = 0
    y = y_test_or_train.values
    for i in range(m):
        sum += np.abs( y_pred[i] - y[i] )
    return (1/m) * sum
