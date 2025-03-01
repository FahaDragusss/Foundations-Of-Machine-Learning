import pandas as pd

def gradient_descent(theta,X,y_pred,target,lr):
    for j in range(len(theta)):
        sum = 0
        for i in range(len(X)):
            sum += (target[i] - y_pred[i])*X[i][j]
        theta[j] = theta[j] - (lr*(sum))/len(target)
    return theta
        