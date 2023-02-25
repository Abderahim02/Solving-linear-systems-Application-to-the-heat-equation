import numpy as np 


### Décomposition de Cholesky ###

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


    


### Méthode du gradient conjugué ###

def conjgrad(A, b, x):
    n = len(b)
    A_cmp = np.zeros((n, n))

    

    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)

    for i in range(n):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < 1e-10:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x




    
    
