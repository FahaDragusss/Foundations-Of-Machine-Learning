import pandas as pd

def gradient_descent(theta , X , y_pred , y_test_or_train , lr):
    for j in range(len(theta)):
        sum = 0
        for i in range(len(X)):
            sum += (y_test_or_train[i] - y_pred[i])*X[i][j]
        theta[j] = theta[j] - (lr*(sum))/len(y_test_or_train)
    return theta
        