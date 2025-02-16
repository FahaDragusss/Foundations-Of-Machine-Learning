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

# Real dataset from kaggle, Feature 1 = Car selling price : Feature 2 = Car manufacturing year
# It is true that there were many more features aswell but for our purposes we can ignore them.
# The Dataset is collected from a Indian car selling website and downloaded from Kaggle.

X = [
    [0.7, 2007], [0.5, 2007], [1.0, 2012], [0.46, 2017], [1.41, 2014],
    [1.25, 2007], [0.25, 2016], [0.6, 2014], [0.25, 2015], [0.78, 2017],
    [0.35, 2015], [1.0, 2014], [0.25, 2018], [0.24, 2015], [0.05, 2019],
    [0.33, 2013], [0.28, 2014], [0.59, 2013], [0.045, 2018], [1.759, 2011],
    [0.145, 2018], [0.15, 2018], [0.5, 2013], [0.338, 2012], [1.304, 2011],
    [0.5, 2016], [0.8, 2015], [0.1, 2019], [1.19, 2010], [0.6, 2014],
    [0.758, 2013], [0.78, 2009], [0.4, 2012], [0.5, 2014], [0.74, 2014],
    [0.64, 2014], [0.6, 2012], [1.2, 2009], [0.79, 2009], [0.15, 2019],
    [0.8, 2006], [0.185, 2017], [0.102, 2018], [0.29, 2018], [0.28, 2018],
    [0.46, 2016], [0.7, 2014], [0.6, 2012], [0.35, 1996], [0.9, 2014],
    [0.8, 2013], [0.8, 2005], [0.733, 2014], [0.92, 2014], [0.66764, 2009],
    [1.0, 2009], [3.5, 2010], [2.3, 2011], [0.6, 2017], [0.31, 2018],
    [0.39, 2007], [1.2, 2009], [1.66, 2014], [1.1, 2006], [0.35, 2017],
    [0.6, 2018], [0.54, 2007], [0.63, 2010], [1.2, 2014], [1.2, 2005],
    [0.76, 2014], [0.8, 2015], [0.25, 2019], [0.11958, 2016], [0.2, 2015],
    [0.09, 2017], [0.065, 2017], [0.7, 2015]
]

year_of_advert = 2022
X = [[km*10, (year_of_advert - year)] for km, year in X]


theta = [0.1,0.1,0.1]  # starting theta

# theta = [7.3907696687093365, -0.15364138788362225, -0.050766029306350074] 
# Theta after first run of the model
# converged in 2255 iterations

#
#   Outlier datapoint Target = 81.5 | Feature = [1.2 , 2013]
#

# Target train

target = [
    0.6, 1.35, 6.0, 2.5, 4.5, 1.4, 5.5, 2.4, 8.5, 3.65,
    2.6, 2.5, 16.5, 5.85, 11.95, 3.9, 19.65, 14.25, 9.75, 11.9,
    9.3, 5.25, 17.35, 13.75, 4.5, 9.0, 13.0, 14.0, 8.5, 2.3,
    15.5, 12.5, 6.25, 10.5, 5.6, 2.9, 4.11, 1.5, 5.0, 1.0,
    7.25, 4.01, 7.5, 3.1, 6.65, 4.65, 1.6, 2.5, 6.75, 3.0,
    0.7, 2.4, 5.25, 1.51, 1.4, 2.8, 3.5, 5.7, 3.0, 1.0,
    5.0, 1.25, 1.3, 9.25, 7.5, 2.0, 2.48, 4.5, 0.8, 6.5,
    4.5, 6.5, 4.95, 3.71, 10.25
]



# X test

#X = [
#    [58000, 2007], [62200, 2013], [34000, 2014], [53000, 2015], [49000, 2013],
#    [15000, 2017], [50000, 2013], [53000, 2013], [15000, 2018], [42000, 2012],
#    [70000, 2015], [110000, 2004], [131365, 2014], [48980, 2016], [46000, 2014],
#    [60000, 2009], [58000, 2015], [22000, 2016], [37000, 2017], [70000, 2009]
#]

# Target test
#y_test = [
#    280000, 370000, 459999, 42000, 42000, 190000, 165000, 710000, 1200000, 229999,
#    310000, 320000, 100000, 160000, 60000, 100000, 350000, 700000, 300000, 830000
#]

#------------------------------------------------------------------
#------------------------- OPTIMAZATION ---------------------------
#------------------------------------------------------------------


# Adding Feature zero X0 
X = add_bias_column(X)
# Making theta a (n+1 x 1) list of lists (Matrix)
theta = one_d_to_two_d(theta)

#print(f'This is X : ', X)
print(f'This is theta : ',theta)

# Compute initial predictions

#y_pred = predict(X, theta)
#print(f'This is y_pred : ',y_pred)

#cost = cost_function(y_pred, target)
#print(f"This is the cost funtion for the first predictions", cost)




# Debugging Gradient descent
# Perform one Gradient Descent step
alpha = 0.0125  # Learning rate
iterations = 0

while iterations != 10000:
    iterations +=1
    y_pred = predict(X,theta)
    theta = batch_gradient_descent(theta, X, y_pred, target, alpha)
    if iterations % 10 == 0:
        print(f'Updated Theta : ', theta)
        print('Cost function : ', cost_function(y_pred,target))

print(f"Final Theta : ", theta)

# Convergence threshold
#tolerance = 1e-6

# Initialize cost as infinity
#previous_cost = float('inf') # cost for first iteration
#previous_cost = 9.426075446978972 # cost after first run of the model
#alpha = 0.0125  # Learning rate
#iteration = 0

# List to store cost values every 10 iterations
#cost_history = []

# Batch Gradient descent
#while True:
#    y_pred = predict(X, theta)  # Compute predictions
#    cost = cost_function(y_pred, target)  # Compute cost
#    
#    if abs(previous_cost - cost) < tolerance:  # Check for convergence
#        break  # Stop when cost change is very small
#
#    theta = batch_gradient_descent(theta, X, y_pred, target, alpha)  # Update theta
#    previous_cost = cost  # Update cost for next iteration
#    iteration += 1
#
#    if iteration % 100 == 0:  # Store cost every 100 iterations
#       cost_history.append(cost)

#print("Final Theta:", theta)
#print("Final Cost:", cost)
#print("Cost function decrease over time : ", cost_history)
#print(f"Converged in {iteration} iterations!")

