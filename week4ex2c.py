# MATH.APP.720 Week 4 exercise 2c:
# Explore the convergence of the function using Matlab or the OptEdu package.

from optedu.problems.rosenbrock import Rosenbrock
from optedu.algorithms.linesearch import backtracking_armijo
import numpy as np

# init Rosenbrock problem
ros = Rosenbrock(a = 1, b = 100, n=2)

# GD params
xk = np.array(([-1.2, 1.0]), dtype = float) # initial point
max_iter = 10000 # iterations of xk
tol = 1e-4 # tolerance for stopping

history = []

for k in range(max_iter):
    grad_fk = ros.grad(xk)
    grad_norm = np.linalg.norm(grad_fk)
    fk = ros.f(xk)

    history.append((fk, grad_norm))

    if grad_norm < tol:
        print(f"Converged in {k} iterations.") # with current tol and iters, we get convergence in 8058 iters
        break

    pk = -grad_fk

    t, neval = backtracking_armijo(ros.f, ros.grad, xk, pk)

    # update iter
    xk = xk + t*pk

print(f"Final solution:", xk) # we get [0.97277594 0.9460233 ]
print(f"Function value:", ros.f(xk)) # we get 0.0007484247354819688

 
