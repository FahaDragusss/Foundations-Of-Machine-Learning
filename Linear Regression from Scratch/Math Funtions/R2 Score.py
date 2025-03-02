import numpy as np

def r2_score(y_pred , y_test_or_train):
    numerator = 0
    denominator = 0
    sum = 0
    m = len(y_pred)
    y = y_test_or_train.values

    for i in range(m):
        sum += y[i]
        numerator += (y_pred[i] - y[i])**2
    mean = sum/m

    for i in range(m):
        denominator += (y[i] - mean)**2
    
    return 1 - (numerator/denominator)

