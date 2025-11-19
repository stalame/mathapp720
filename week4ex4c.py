# MATH.APP.720 Week 4 exercise 4c: 
# Starting from x0 = (0.2, 0.2), 
# take one iteration with τ = 1 Report the new point and
# function value. (By hand, Matlab or OptEdu)

from optedu.optedu.algorithms.newton import newton
from optedu.optedu.algorithms.newton import _pd_safeguard
import numpy as np

def f(X):
    x, y = X
    return x**4 + y**4 - 3*x**2 -3*y**2 + 2*x*y


def grad(X):
    x, y = X
    return np.array([4*x**3 - 6*x + 2*y,
                     4*y**3 - 6*y + 2*x])

def hess(X):
    x, y = X
    return np.array([[12*x**2 - 6, 2],
                     [2, 12*y**2 - 6]])

def modified_hess(X):
    tau = 1
    G = hess(X) + tau * np.eye(2)
    return _pd_safeguard(G, eps=1) # 1 iteration


x0 = np.array([0.2, 0.2], dtype=float)

result = newton(f=f, grad=grad, hess=modified_hess, x0=x0, tol=1e-8, maxit=1, safeguard=False)

print("Starding point:", x0) # here we have x0=[0.2, 0.2]
print("New point:", result['x']) # we get x1=[0.968, 0.968]
print("Function value at new point:", result['f']) # we get f(x1) = -1.99206...