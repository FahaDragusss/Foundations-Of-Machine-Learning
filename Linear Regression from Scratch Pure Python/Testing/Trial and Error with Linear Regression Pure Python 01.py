# Initializing all relevant funtions

def add_bias_column(X):
    return[[1] + row for row in X]

def one_d_to_two_d(theta):
    return [[t] for t in theta]

def predict(X,theta):
    return matrix_multiplication(X,theta)

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

# Initialize Data
X = [
    [2, 3],
    [4, 5],
    [6, 7]
]
y_true = [10, 20, 30]
theta = [0.1, 0.2, 0.3]

# Convert data
X = add_bias_column(X)
theta = one_d_to_two_d(theta)

print(f'This is X : ', X)
print(f'This is theta : ',theta)

# Compute initial predictions and cost
y_pred = predict(X, theta)

print(f'This is y_pred : ',y_pred)

cost = cost_function(y_pred, y_true)

print("Initial Cost : ", cost)

# Perform one Gradient Descent step
alpha = 0.02  # Learning rate
theta_new = batch_gradient_descent(theta, X, y_pred, y_true, alpha)
print("Updated Theta : ", theta_new)