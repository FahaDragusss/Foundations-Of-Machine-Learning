import numpy as np

def mean_squared_error(y_pred, y_test_or_train):
    m = len(y_pred)
    sum = 0
    y = y_test_or_train.values
    for i in range(m):
        sum += ( y_pred[i] - y[i] )**2
    return ( 1 / m ) * sum

