def matrix_2x2_inverse(A):

    a = A[0][0]
    b = A[0][1]
    c = A[1][0]
    d = A[1][1] 
    
    y = (a*d)-(b*c)
    
    if y == 0:
        raise ValueError(f"Inverse not possible as Determinant is zero")

    x = 1/(y)
    
    B = [[d*x,-b*x],[-c*x,a*x]]

    return B