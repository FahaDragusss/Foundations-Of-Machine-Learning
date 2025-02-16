def r2_score(target,y_pred):
    total_error_squared = 0
    total_distance_from_mean_squared = 0
    total_sum = 0
    m = len(target)
    
    #Calculate mean
    for i in range(m):
        total_sum += target[i]
    mean = total_sum/m

    #Calculate r2 score
    for i in range(m):
        total_error_squared += (target[i] - y_pred[i][0])**2
        total_distance_from_mean_squared += (target[i] - mean)**2

    if total_distance_from_mean_squared == 0:
        return 0.0
    
    return 1 - (total_error_squared/total_distance_from_mean_squared)


    