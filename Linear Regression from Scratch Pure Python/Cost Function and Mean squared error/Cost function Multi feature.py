def cost_function(y_pred,target):
    m = len(target)
    total_error = 0
    for i in range(m):
        total_error += (y_pred[i][0]-target[i])**2
    return ( 1/(2*m) * (total_error))