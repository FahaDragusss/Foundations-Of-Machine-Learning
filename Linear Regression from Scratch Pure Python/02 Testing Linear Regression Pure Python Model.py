#------------------------------------------------------------------
#------------------------- FUNCTIONS ------------------------------
#------------------------------------------------------------------

# Adding bias column of X0 = 1 to normalize our equation of y_pred
def add_bias_column(X): 
    return[[1] + row for row in X]

# Converting theta from a list to a list of lists with 1 element (Making is into a n+1 x 1 matrix)
def one_d_to_two_d(theta): 
    return [[t] for t in theta]

# Calculating y_pred
def predict(X,theta):
    return matrix_multiplication(X,theta)

# Calculating Cost
def cost_function(y_pred,target):
    m = len(target)
    total_error = 0
    for i in range(m):
        total_error += (y_pred[i][0]-target[i])**2
    return ( 1/(2*m) * (total_error))

def batch_gradient_descent(theta,X,y_pred,target,lr):
    #Theta already converted to 2D column
    for j in range(len(theta)):
        sum = 0
        for i in range(len(target)):
                sum += (y_pred[i][0]-target[i])*X[i][j]
        theta[j][0] = theta[j][0] - (lr*(sum))/len(target)
    return theta

# Matrix Operations
def matrix_multiplication(A,B):
    
    A_row = len(A)
    B_col = len(B[0])
    A_col = len(A[0])
    B_row = len(B)

    if A_col != B_row:
        raise ValueError(f"Matrix multiplication not possible: A has {A_col} columns, but B has {B_row} rows.")
    
    C = [[0 for i in range(B_col)] for j in range(A_row)]
    
    for i in range(A_row):
        for j in range(B_col):
            for k in range(A_col):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Math Functions
def mean_absolute_error(target,y_pred):
    m = len(target)
    total_error = 0
    for i in range(m):
        total_error += abs(target[i]-y_pred[i][0])
    return (1/m)*total_error

def mean_squared_error(target,y_pred):
    m = len(target)
    total_error = 0
    for i in range(m):
        total_error += (target[i]-y_pred[i][0])**2
    return (1/m)*total_error

def root_mean_squared_error(mse):
    return mse**0.5

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

#------------------------------------------------------------------
#--------------------------- DATASET ------------------------------
#------------------------------------------------------------------

# X test

X = [
    [58000, 2007], [62200, 2013], [34000, 2014], [53000, 2015], [49000, 2013],
    [15000, 2017], [50000, 2013], [53000, 2013], [15000, 2018], [42000, 2012],
    [70000, 2015], [110000, 2004], [131365, 2014], [48980, 2016], [46000, 2014],
    [60000, 2009], [58000, 2015], [22000, 2016], [37000, 2017], [70000, 2009]
]

year_of_advert = 2022
X = [[km/10000, (year_of_advert - year)] for km, year in X]
X = add_bias_column(X)

# Target test
target = [
    2.80000, 3.70000, 4.59999, .42000, .42000, 1.90000, 1.65000, 7.10000, 12.00000, 2.29999,
    3.10000, 3.20000, 1.00000, 1.60000, .60000, 1.00000, 3.50000, 7.00000, 3.00000, 8.30000
]

#------------------------------------------------------------------
#---------------------- TESTING THE MODEL -------------------------
#------------------------------------------------------------------

theta = [[7.3907696687093365], [-0.15364138788362225], [-0.050766029306350074]]

y_pred = predict(X,theta)

print(f"Cost : ", cost_function(y_pred,target))
print(f"Mean squared error : ", mean_squared_error(target,y_pred))
print(f"Root mean squared error : ", root_mean_squared_error(mean_squared_error(target,y_pred)))
print(f"Mean absolute error : ", mean_absolute_error(target,y_pred))
print(f"R2 score : ", r2_score(target,y_pred))
print(f"These are the model predictions : ", y_pred)