def mean_squared_error(b,w,y_true,x,m):

    mse = 0

    for i in range(m):
        y_pred = b + (w*x[i])
        mse += (y_pred-y_true[i])**2
    
    return mse/m


