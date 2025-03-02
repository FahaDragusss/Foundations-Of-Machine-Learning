import numpy as np

def cost_function(y_pred, y_test_or_train):
    m = len(y_pred)
    sum = 0
    y = y_test_or_train.to_numpy()
    for i in range(m):
        sum += ( y_pred[i] - y[i] )**2
    return ( 1 / (2 * m) ) * sum
