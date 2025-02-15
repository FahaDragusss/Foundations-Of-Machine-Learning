#------------------------------------------------------------------
#------------ THE LINEAR REGRESSION PURE PYTHON -------------------
#------------------------------------------------------------------


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
        theta[j][0] = theta[j][0] - lr*(sum)
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


