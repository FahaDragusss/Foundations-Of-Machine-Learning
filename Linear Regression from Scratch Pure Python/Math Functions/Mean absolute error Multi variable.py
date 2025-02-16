def mean_absolute_error(target,y_pred):
    m = len(target)
    total_error = 0
    for i in range(m):
        total_error += abs(target[i]-y_pred[i][0])
    return (1/m)*total_error