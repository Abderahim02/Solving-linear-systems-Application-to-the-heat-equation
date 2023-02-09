import numpy as np 

def cholesky(A):
    n=A.shape[0]
    T=np.zeros((n,n))
    for i in range(0, n ,1):
        s=0
        for k in range (0, i): ##remplir la diagonale 
            s+=T[i][k]
        T[i][i] = np.sqrt(A[i][i] - s)
        for j in range (i, n): ## remplir la partie inferieure
            t=0
            for k in range (0, i):
                t+=T[i][k] * T[j][k]
            T[j][i]=(A[i][j] - t)/A[i][i]
    return T


def test_chloesky():
    A=np.array([[1,1,1,1],
                [1,5,5,5],
                [1,5,14,14],
                [1,5,14,15]])
    print(cholesky(A))

test_chloesky()
    
