def mean_squared_error(target,y_pred):
    m = len(target)
    total_error = 0
    for i in range(m):
        total_error += (target[i]-y_pred[i][0])**2
    return (1/m)*total_error