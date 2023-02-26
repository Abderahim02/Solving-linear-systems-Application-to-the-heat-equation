import numpy as np 
import random  

### 1st partie  ## 
### ##Décomposition de Cholesky ##########################################
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
  
####Décomposition incomplete de Cholesky ###########################################
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

######Génération de matrices symetriques positives #################################""""
def matrix_sn_generator(m, n):
    max_value = 100 #la valeur maximale qu'un coefficient peut prendre 
    L = np.zeros((n,n)) #matrice nulle
    for i in range(n):  ##on remplit d'abord la diagonale aléatoirement
        L[i][i]=float(random.randint(1, max_value)) 
    i,j,added_coeff = 0, 0, n #added_coeff est le nombre de coefficents non nuls ajoutés
    for i in range (n):
        for j in range (i+1, n):
            if ( i != j):
                if(added_coeff < m ): #on remplit les elemtns extra diagonaux par des valeurs 
                                      #tq leur somme est toujours inferieur au coefficent diagonal 
                        L[i][j] = L[j][i] = float(random.randint(1, L[i][i] ) / n) 
                        added_coeff+=2 #on ajoute par paquets de 2
                else: #lorsque on atteint le nombre de coeff non nuls demandés on retourne la matrice
                    return L
    return L

def is_dominant_diagonal_matrix(matrix): #une fonction pour le tests
    n, m = np.shape(matrix)
    if n != m:
        return False
    if not np.allclose(matrix, matrix.T):
        return False

    for i in range (n):
        somme=0
        diag_val = matrix[i][i] 
        for j in range (n):
            somme += matrix[i][j] 
        print(somme)
        row_sum = somme - diag_val
        if diag_val <= row_sum:
            return False
    return True

def test_matrix_sn_generator():
    print(matrix_sn_generator(14, 4))
    print(is_dominant_diagonal_matrix(matrix_sn_generator(14, 4)))
    print("\n")
test_matrix_sn_generator()



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




