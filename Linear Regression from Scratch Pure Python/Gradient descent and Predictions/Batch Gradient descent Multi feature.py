def batch_gradient_descent(theta,X,y_pred,target,lr):
    #Theta already converted to 2D column
    for j in range(len(theta)):
        sum = 0
        for i in range(len(target)):
                sum += (y_pred[i][0]-target[i])*X[i][j]
        theta[j][0] = theta[j][0] - lr*(sum)
    return theta