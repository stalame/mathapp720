# MATH.APP.720 Week 4 exercise 3d: 
# Write the Newton step sk = −H(xk)
# −1∇f(xk) and update xk+1 = xk + sk. Starting
# from x0 = (0.2, 0.2), show numerically or symbolically that Newton may diverge or
# get attracted toward the saddle at (0, 0). (By hand, using Matlab, or OptEdu)

from optedu.optedu.algorithms.newton import newton
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

# initial point near saddle (0,0)
x0_list = [
    np.array([0.2, 0.2], dtype=float),
    np.array([0.01, -0.01], dtype=float),
    np.array([-0.5, 0.5], dtype=float)
]

for x0 in x0_list:
    result = newton(f=f, grad=grad, hess=hess, x0=x0, tol=1e-8, maxit=100)
    print("Starting point:", x0) # here we have whatever x0 in x0_list, see above
    print("Status:", result['status']) # we get 'converged' in all cases

    print("Final x:", result['x'])
    # Final x values:
    # for x0=[0.2, 0.2] we get [-3.485e-15, -3.485e-15]
    # for x0=[0.01, -0.01] we get [1.00e-18, -1.00e-18]
    # for x0=[-0.5, 0.5] we get [-2.34e-19, 2.34e-19]

    print("Final f:", result['f'])
    # Final f values:
    # for x0=[0.2, 0.2], f(x0) = -4.8568790882322527e-29
    # for x0=[0.01, -0.01], f(x0) = -8.007230314965692e-36
    # for x0=[-0.5, 0.5] f(x0) = -4.394097445069082e-37

    print("Iterations:", result['counts']['nit'])
    # Iterations:
    # for x0=[0.2, 0.2] we have 3 iters
    # for x0=[0.01, -0.01] we have 2 iters
    # for x0=[-0.5, 0.5] we have 4 iters