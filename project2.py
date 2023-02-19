import numpy as np 

def cholesky(A):
    n = len(A)
    L = np.zeros((n,n))
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if (i == k):
                L[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1 / L[k][k] ) * (A[i][k] - tmp_sum)
    return L

def test_chloesky():
    A=np.array([[1,1,1,1],
                [1,5,5,5],
                [1,5,14,14],
                [1,5,14,15]])

    B=np.array([[1,1,1,1],
                [1,2,2,2],
                [1,2,3,3],
                [1,2,3,4]])
    print(cholesky(A))
    print("\n")
    print(cholesky(B))

test_chloesky()
    
def cholesky_incomplete(A):
    n = len(A)
    L = np.zeros((n,n))
    for i in range(n):
        for k in range(i+1):
            if ( A[i][k] != 0):
                tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
                if (i == k):
                    L[i][k] = np.sqrt(A[i][i] - tmp_sum)
                else:
                    L[i][k] = (1 / L[k][k] ) * (A[i][k] - tmp_sum)
    return L
