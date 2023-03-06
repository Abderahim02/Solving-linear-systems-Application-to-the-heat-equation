import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
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
def f_uniform(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
def f_center(x, y):
    # Distance du centre
    r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    # Temperature elevée au centre mais decroit loin de la source
    f = np.where(r <= 0.2, np.exp(-10*(r-0.25)**2), 0)
    # intensification de la chaleur
    f *= 5
    return f
def f_linear(x, y):
    T_left = 2
    T_right = 0.0
    f = T_left - (T_left - T_right) * x
    return f




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
    T=conjgrad(A, b,np.zeros(len(b)))
    return T


def f(x, y):
    return np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1)
#Generate temperature field using the found T

N = 60 # Grid size
T_uniform = find_A_and_b(N, f_uniform).reshape((N, N))
#Plot temperature as a heatmap

fig, ax = plt.subplots()
im = ax.imshow(T_uniform, cmap='hot', vmin=0, vmax=1)
ax.set_title('Temperature cas flux uniforme (gradient conjugué)')
fig.colorbar(im)
plt.show()


T_center = find_A_and_b(N, f_center).reshape((N, N))
#Plot temperature as a heatmap

fig, ax = plt.subplots()
im = ax.imshow(T_center, cmap='hot', vmin=0, vmax=1)
ax.set_title('Temperature cas flux centre (gradient conjugué)')
fig.colorbar(im)
plt.show()



T_linear = find_A_and_b(N, f_linear).reshape((N, N))
#Plot temperature as a heatmap

fig, ax = plt.subplots()
im = ax.imshow(T_linear, cmap='hot', vmin=0, vmax=1)
ax.set_title('Temperature cas flux Décroissant (gradient conjugué)')
fig.colorbar(im)
plt.show()


def find_A_and_b_1(N, f):
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
    T = solve(A, b)
    return T




# Generate temperature field using the found T


N = 60 # Grid size
T_1_uniform = find_A_and_b_1(N, f_uniform).reshape((N, N))
#Plot temperature as a heatmap

fig, ax = plt.subplots()
im = ax.imshow(T_1_uniform, cmap='hot', vmin=0, vmax=1)
ax.set_title('Temperature cas flux uniforme obtenue avec résolution de numpy')
fig.colorbar(im)
plt.show()


T_1_center = find_A_and_b_1(N, f_center).reshape((N, N))
#Plot temperature as a heatmap

fig, ax = plt.subplots()
im = ax.imshow(T_1_center, cmap='hot', vmin=0, vmax=1)
ax.set_title('Temperature cas flux centre obtenue avec résolution de numpy')
fig.colorbar(im)
plt.show()



T_linear_1 = find_A_and_b_1(N, f_linear).reshape((N, N))
#Plot temperature as a heatmap

fig, ax = plt.subplots()
im = ax.imshow(T_linear_1, cmap='hot', vmin=0, vmax=1)
ax.set_title('Temperature cas flux Décroissant obtenue avec résolution de numpy')
fig.colorbar(im)
plt.show()
