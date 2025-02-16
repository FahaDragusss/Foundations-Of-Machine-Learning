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

# Calculating cost function
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

# Target test
target = [
    280000, 370000, 459999, 42000, 42000, 190000, 165000, 710000, 1200000, 229999,
    310000, 320000, 100000, 160000, 60000, 100000, 350000, 700000, 300000, 830000
]

target = [y/100000 for y in target ]

#------------------------------------------------------------------
#---------------------- TESTING THE MODEL -------------------------
#------------------------------------------------------------------

theta = [[7.3907696687093365], [-0.15364138788362225], [-0.050766029306350074]]

predict(X,theta)