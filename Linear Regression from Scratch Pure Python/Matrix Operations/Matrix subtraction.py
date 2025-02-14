def matrix_subtraction(A,B):
    return [[A[i][j]-B[i][j] for j in range (len(A[i]))] for i in range (len(A))]