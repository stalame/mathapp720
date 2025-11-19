# MATH.APP.720 Week 4 exercise 1c:
# Repeat with backtracking line search (Armijo) with parameters c = 10−4 and ρ = 0.5;
# report the accepted αk and x1.

from optedu.algorithms.linesearch import backtracking_armijo
import numpy as np

def f(X):
    x,y = X

    Q = np.array([[2, 0],[0, 20]], dtype=float)
    b = np.array([-4,20], dtype=float)

    c = 14

    return float(0.5 * X.T @ Q @ X + b.T @ X + c)

def grad_f(X):
    x,y = X
    Q = np.array([[2, 0],[0, 20]], dtype=float)
    b = np.array([-4,20], dtype=float)

    return Q @ X + b

x0 = np.array([0,0], dtype = float)

p0 = -grad_f(x0)

t, neval = backtracking_armijo(f, grad_f, x0, p0)

print("Step size:", t) # we get 0.0625
print("Value of f:", f(x0+t*p0)) # we get 3.6875