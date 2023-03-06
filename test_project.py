import project2
import numpy as np

### Méthode du gradient conjugué ###

A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
x = np.zeros(len(b))
x = project2.conjgrad(A, b, x)

print("Solution:", x)
