#%%
import numpy as np 
import random  
import matplotlib.pyplot as plt
### 1st partie  ## 
### ##Décomposition de Cholesky ##########################################
def cholesky(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
            if (i == k):
                L[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1 / L[k][k]) * (A[i][k] - tmp_sum)
    return L


def test_chloesky():
    A = np.array([[1, 1, 1, 1],
                  [1, 5, 5, 5],
                  [1, 5, 14, 14],
                  [1, 5, 14, 15]])

    cA = np.linalg.cholesky(A)

    B = np.array([[1, 1, 1, 1],
                  [1, 2, 2, 2],
                  [1, 2, 3, 3],
                  [1, 2, 3, 4]])

    cB = np.linalg.cholesky(B)

    print(cholesky(A))
    print(cA)

    print("\n")
    print(cholesky(B))
    print(cB)


test_chloesky()

# %% ###Décomposition incomplete de Cholesky ###########################################


def cholesky_incomplete(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(i+1):
            if (A[i][k] != 0):
                tmp_sum = sum(L[i][j] * L[k][j] for j in range(k))
                if (i == k):
                    L[i][k] = np.sqrt(A[i][i] - tmp_sum)
                else:
                    L[i][k] = (1 / L[k][k]) * (A[i][k] - tmp_sum)
    return L

# %% #####Génération de matrices symetriques positives #################################""""


def matrix_sn_generator(m, n):
    max_value = 5 #la valeur maximale qu'un coefficient peut prendre 
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


def test_chloesky_incomplete(m, n):
    test_matrix = matrix_sn_generator(m, n)
    chol_exp = np.linalg.cholesky(test_matrix)
    chol_out = cholesky_incomplete(test_matrix)
    print(chol_exp)
    print(chol_out)
    return np.allclose(chol_exp, chol_out)


def is_dominant_diagonal_matrix(matrix):  # une fonction pour le tests
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
        if (diag_val <= row_sum):
            return False
    return True

def test_matrix_sn_generator():
    matrix = matrix_sn_generator(14, 4)
    print(matrix)
    print(is_dominant_diagonal_matrix(matrix))
    print("\n")


test_matrix_sn_generator()


## préconditionneur
def is_good_conditionner_complete(A):
    T = cholesky(A)
    T_inv= np.linalg.inv(T)
    return np.linalg.cond(np.dot(np.transpose(T_inv), T_inv )) < np.linalg.cond(A)
         
def test_is_conditionner_complete():
    A=np.array([[1,1,1,1],
                [1,5,5,5],
                [1,5,14,14],
                [1,5,14,15]])
    print(is_good_conditionner_complete(A))
test_is_conditionner_complete()
### Méthode du gradient conjugué ###

def conjgrad(A, b, x):
    n = len(b)
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r.T, r)
    for i in range(n):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p.T, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.T, r)
        if np.sqrt(rsnew) < 1e-10:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def conjgrad_prec(A, b, x, P, nmax, tol):
    flag = 0
    iter = 0
    bnrm2 = np.linalg.norm(b)
    if bnrm2 == 0:
        bnrm2 = 1
    r = b - np.dot(A, x)
    relres = []
    relres1 = np.linalg.norm(r) / bnrm2
    relres.append(np.linalg.norm(r) / bnrm2)
    if relres1 < tol:
        return
    for iter in range(nmax):
        z = np.linalg.solve(P, r)
        rho = np.dot(r.T, z)
        if iter > 1:
            beta = rho / rho1
            p = z + beta * p
        else:
            p = z
        q = np.dot(A, p)
        alpha = rho / np.dot(p.T, q)
        x = x + alpha * p
        r = r - alpha * q
        relres1 = np.linalg.norm(r) / bnrm2
        relres.append(np.linalg.norm(r) / bnrm2)
        if relres1 <= tol:
            break
        rho1 = rho
    if relres1 > tol:
        flag = 1
    return x, relres, iter, flag

##simulation..............................
def vector_generator(n):
    max_value = 5
    return np.array([ random.randint(1, max_value) for i in range (n)])

def matrix_generator(n):
    max_value = 5
    return np.array([ [random.randint(1, max_value) for i in range (n)] for j in range (n)])


def calcul_normeA(y, A):
    return np.dot( np.dot(np.transpose(y), A), y)

def conjgrad_simulation(A, b, x, nmax):
    n = len(b)
    A_cmp = np.zeros((n, n))
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    simulation_vector_norm1 = []
    simulation_vector_norm2 = []
    simulation_vector_normA = []
    theorique_solution = np.linalg.solve(A,b)
    for i in range(nmax):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        #norme 1 
        tmp_diff_1 = np.linalg.norm(theorique_solution - x, ord=1)
        norme_x_1 =  np.linalg.norm( x, ord = 1)
        #norme 2
        tmp_diff_2 = np.linalg.norm(theorique_solution - x)
        norme_x_2 =  np.linalg.norm(x)
        #norme definie par la matrice A appartenat à Sn+  
        tmp_diff_A = calcul_normeA(theorique_solution - x, A)
        norme_x_A =  calcul_normeA(x, A)
        #stockage des valeurs 
        simulation_vector_norm1.append( tmp_diff_1/norme_x_1)
        simulation_vector_norm2.append( tmp_diff_2/norme_x_2)
        simulation_vector_normA.append( tmp_diff_A/norme_x_A)
        if np.sqrt(rsnew) < 1e-10:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return [x, simulation_vector_norm1, simulation_vector_norm2, simulation_vector_normA]


def conjugate_gradient_for_compare(A, b, x, max_iterations, tol):
    flag = 0
    iteration = 0
    bnrm2 = np.linalg.norm(b)
    if bnrm2 == 0:
        bnrm2 = 1
    r = b - np.dot(A, x)
    relres = []
    relres1 = np.linalg.norm(r) / bnrm2
    relres.append(np.linalg.norm(r) / bnrm2)
    if relres1 < tol:
        return
    for iteration in range(max_iterations):
        z = r
        rho = np.dot(r.T, z)
        if iteration > 1:
            beta = rho / rho1
            p = z + beta * p
        else:
            p = z
        q = np.dot(A, p)
        alpha = rho / np.dot(p.T, q)
        x = x + alpha * p
        r = r - alpha * q
        relres1 = np.linalg.norm(r) / bnrm2
        relres.append(relres1)
        if relres1 <= tol:
            break
        rho1 = rho
    if relres1 > tol:
        flag = 1
    return x, relres, iteration, flag


def simulation_conjrad_norm1(n):
    np.random.seed(100)
    A = matrix_sn_generator(n/2, n)
    x_random = vector_generator(n)
    b = vector_generator(n)
    solution = conjgrad_simulation(A,b,x_random, 40)
    x = np.linspace(0, len(solution[1]), len(solution[1]))
    plt.plot(x,solution[1], label= 'avec n = '+ str(n))
    plt.xlabel("Le nombre ditérations")
    plt.ylabel("L'erreur relative de la solution")
    plt.legend()

def simulation_conjrad_norm2(n):
    np.random.seed(100)
    A = matrix_sn_generator(n/2, n)
    x_random = vector_generator(n)
    b = vector_generator(n)
    solution = conjgrad_simulation(A,b,x_random, 40)
    x = np.linspace(0, len(solution[2]), len(solution[2]))
    plt.plot(x,solution[2], label= 'avec n = '+ str(n))
    plt.xlabel("Le nombre ditérations")
    plt.ylabel("L'erreur relative de la solution")
    plt.legend()

def simulation_conjrad_normA(n):
    np.random.seed(100)
    A = matrix_sn_generator(n/2, n)
    x_random = vector_generator(n)
    b = vector_generator(n)
    solution = conjgrad_simulation(A,b,x_random, 40)
    x = np.linspace(0, len(solution[3]), len(solution[3]))
    plt.plot(x,solution[3], label= 'avec n = '+ str(n))
    plt.xlabel("Le nombre ditérations")
    plt.ylabel("L'erreur relative de la solution")
    plt.legend()

def norms_compare(n):
    np.random.seed(100)
    A = matrix_sn_generator(n/2, n)
    x_random = vector_generator(n)
    b = vector_generator(n)
    solution = conjgrad_simulation(A, b, x_random, 40)
    x = np.linspace(0, len(solution[1]), len(solution[1]))
    plt.plot(x,solution[1], label= 'avec la norme 1')
    plt.plot(x,solution[2], label= 'avec la norme 2')
    plt.plot(x,solution[3], label= 'avec la norme définie par A')
    plt.xlabel("Le nombre ditérations")
    plt.ylabel("L'erreur relative de la solution")
    plt.legend()
    plt.show()
norms_compare(40)

def conditionner_generator(A):
    T = cholesky(A)
    T_inv= np.linalg.inv(T)
    return np.dot(np.transpose(T_inv), T_inv )
def conj_compare(n):
    np.random.seed(5)
    A = matrix_sn_generator(n/2, n)
    x_random = vector_generator(n)
    b = vector_generator(n)
    M = conditionner_generator(A)

    solution = conjugate_gradient_for_compare(A, b, x_random, 50, 1e-10)
    solution_prec = conjgrad_prec(A, b, x_random, M, 50, 1e-10 )
    x_1 = np.linspace(0, len(solution[1]), len(solution[1]))
    x_2 = np.linspace(0, len(solution_prec[1]), len(solution_prec[1]))
    plt.plot(x_1,solution[1], label= 'avec préconditionnement')
    plt.plot(x_2,solution_prec[1], label= 'sans préconditionnement')
    plt.xlabel("Le nombre ditérations")
    plt.ylabel("La valeur du résidus")
    plt.legend()
    plt.show()
conj_compare(40)

def solve_two_dimensional_equation(N, f): #fonction qui prend en entrée N et f fonction caractérisant le flux de la chaleur
    # Construction de la matrice A
    matrice = np.zeros((N**2,N**2))
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            if j == i-1:    
                matrice[i][j] = 1/h**2
            if j == i-N:
                matrice[i][j] = 1/h**2
            if j==i:
                matrice[i][j] = -4/h**2
            if j==i+1:
                matrice[i][j] = 1/h**2
            if j==i+N:
                matrice[i][j] = 1/h**2      
    # Construction du vecteur b
    # Utilisation de la méthode de différences finis pour discrétisation de l'espace
    #dans ce cas on va modéliser le problème dans un carré [0;1]×[0,1]
######TROISIEME PARTIE #################################""""
def find_A_and_b(N, f):
    # N*N désigne le nombre de points de la discrétisation spatiale
    # f désigne la fonction f:(x,y)->f(x,y) représentant le flux
    # Utilisation de la méthode de différences finies pour discrétisation de l'espace
    # dans ce cas on va modéliser le problème dans un carré [0;1]×[0,1]
    h = 1 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    y = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Construction de la matrice A
    A = np.zeros((N**2, N**2))
    for i in range(len(A)):
        for j in range(len(A)):
            if j == i - 1:
                A[i][j] = 1 / h**2
            if j == i - N:
                A[i][j] = 1 / h**2
            if j == i:
                A[i][j] = -4 / h**2
            if j == i + 1:
                A[i][j] = 1 / h**2
            if j == i + N:
                A[i][j] = 1 / h**2

    # Construction du vecteur b
    b = -f(X, Y)
    b = b.flatten() #on transforme cette grille de N*N elements en un vecteur de N**2 éléments

    # Une fois la matrice A construite et le vecteur b construit, ce n'est plus qu'un système linéaire
    return [A, b]

 
   



# %%
