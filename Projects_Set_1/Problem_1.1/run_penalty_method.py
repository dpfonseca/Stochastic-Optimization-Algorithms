# Home Problem 1.1 - Daniel Peña Fonseca


import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure




# ==============================
# run_gradient_descent function:
# ==============================
def run_gradient_descent(x_start, mu, eta, gradient_tolerance):

    starting_point = np.array(x_start.copy())
    next_point = starting_point - eta * compute_gradient(starting_point, mu)

    maximum_number_of_iterations = 10**6
    i = 0
    while np.linalg.norm(compute_gradient(next_point, mu)) > gradient_tolerance:

        starting_point = next_point.copy()
        next_point = starting_point - eta * compute_gradient(starting_point, mu)

        i += 1
        if i>maximum_number_of_iterations:
            break

    return next_point


# ==============================
# compute_gradient function:
# ==============================
def compute_gradient(x, mu):

    x_1 = x[0]
    x_2 = x[1]

    if x_1**2 + x_2**2 -1 > 0:
        gradient_first_coordinate = 2*(x_1-1) + mu * 2 * (x_1**2 + x_2**2 -1) * 2 * x_1
        gradient_second_coordinate = 4*(x_2-2) + mu * 2 * (x_1**2 + x_2**2 -1) * 2 * x_2

    else:
        gradient_first_coordinate = 2*(x_1-1)
        gradient_second_coordinate = 4*(x_2-2)

    return np.array([gradient_first_coordinate, gradient_second_coordinate])


# ==============================
# Main program:
# ==============================
mu_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 30, 50, 75, 100, 300, 500, 1000, 1200]
eta = 0.0001
x_start = [1,2]
gradient_tolerance = 0.0000001

first_coordinate_convergence =[]
second_coordinate_convergence =[]

for mu in mu_values:
  x = run_gradient_descent(x_start, mu, eta, gradient_tolerance)
  first_coordinate_convergence.append(x[0])
  second_coordinate_convergence.append(x[1])
  output = f"x = ({x[0]:.4f}, {x[1]:.4f}), f(x1,x2) = {((x[0]-1)**2 + 2*((x[1]-2)**2)):.4f}, g(x1,x2) = {((x[0])**2 + ((x[1])**2) - 1):.4f}, mu = {mu:.1f}"
  print(output)



plt.figure(figsize=(3,2), dpi = 300)

plt.semilogx(mu_values, first_coordinate_convergence, 'ko-', linewidth=1, markersize=3, label = r'$x_1^*$')
plt.semilogx(mu_values, second_coordinate_convergence, 'bo-', linewidth=1, markersize=3,label = r'$x_2^*$')
plt.grid()
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel(r'$\mu$', fontsize = 8)
plt.ylabel(r'$x_1^*, x_2^*$', fontsize = 8)
plt.legend(fontsize = 6)
plt.tight_layout()
plt.show()
