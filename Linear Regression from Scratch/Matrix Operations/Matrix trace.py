def matrix_trace(A):
    
    A_row = len(A)
    A_col = len(A[0])

    if A_col != A_row:
        raise ValueError(f"A trace for a non square matrix is not possible.")
    
    trace_A = 0

    for i in range(A_row):
        trace_A += A[i][i]
    return trace_A
