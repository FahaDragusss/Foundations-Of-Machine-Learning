def matrix_transpose(A):

    A_row = len(A)
    A_col = len(A[0])
    A_t = [[0 for i in range(A_row)] for j in range(A_col)]

    for i in range(A_row):
        for j in range(A_col):
            A_t[j][i] = A[i][j]
                
    return A_t