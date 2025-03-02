import pandas as pd
import numpy as np

def batch_gradient_descent(theta , X_train , y_train , lr):
    y_train = y_train.to_numpy()
    y_pred = np.dot(X_train , theta)  

    for j in range(len(theta)):
        sum = 0

        for i in range(len(X_train)):
            sum += (y_train[i] - y_pred[i])*X_train[i, j]
            
        theta[j] = theta[j] - (lr*(sum))/len(y_train)

    return theta
        