from optedu.algorithms import lp_two_phase
import numpy as np


A = np.array([
    [-1, 1], # -x1 +x2 <= 3
    [-2, 1], # -2x1 +x2 <= 2
    [1, 0] # x1 <= 3
    ], dtype = float) 

b = np.array([3, 2, 3], dtype = float)
c = np.array([-1, -2], dtype=float) # maximize cost function

senses = ["le", "le", "le"] # all constraints are less than or equal to inequalities


original_simplex = lp_two_phase.simplex_standard

def simplex_verbose(*args, **kwargs):
    result = original_simplex(*args, **kwargs)
    print()
    print("Intermediate step:")
    if "lp" in result and "basis" in result["lp"]:
        print("  Basis:", result["lp"]["basis"])
    if "x" in result:
        print("  Solution:", result["x"])
    if "f" in result:
        print("  Objective f:", result["f"])
    return result

lp_two_phase.simplex_standard = simplex_verbose

lb = np.zeros(2) # lower bounds for x1 and x2

result = lp_two_phase.solve_two_phase_generic(A, b, c, senses = senses, lb=lb)

x_opt = result.get("x", None)
f_opt = result.get("f", None)
status = result.get("status", None)

if x_opt is not None:
    print("Original variables:", x_opt[:2])
else:
    print("No solution available")

print()
print("Objective value:", f_opt)
print("Status:", status)

#x_opt = result["x"]
#f_opt = result["f"]
#status = result.status["status"]

#print("Optimal solution:", x_opt)
#print("Objective value:", f_opt)
#print("Status:", status)
