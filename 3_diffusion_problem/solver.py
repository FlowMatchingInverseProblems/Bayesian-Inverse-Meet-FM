# -div (\kappa \nabla u) = 0 
# kappa is custom defined 2d grid 
# u(x = 0, y) = f(y, e1) = exp(-0.5 * (y - e1)^2)
#  u(x = 1, y) = g(y, e2) = - exp(-0.5 * (y - e2)^2)

import numpy as np


def solve_pde(kappa, e1=0.5, e2=0.5):
    """
    Solves -div(kappa grad u) = 0 with Dirichlet BCs:
    - u(x=0, y) = exp(-0.5*(y - e1)^2)
    - u(x=1, y) = -exp(-0.5*(y - e2)^2)
    and Neumann BCs (zero flux) at y=0 and y=1.
    """
    Ny, Nx = kappa.shape
    hx = 1.0 / (Nx - 1)
    hy = 1.0 / (Ny - 1)
    total_nodes = Nx * Ny
    A = np.zeros((total_nodes, total_nodes))
    b = np.zeros(total_nodes)
    
    for j in range(Ny):
        y_pos = j * hy
        for i in range(Nx):
            k = i + j * Nx
            if i == 0 or i == Nx - 1:
                # Dirichlet BCs (x=0 and x=1)
                A[k, k] = 1.0
                if i == 0:
                    b[k] = np.exp(-0.5 * (y_pos - e1)**2)
                else:
                    b[k] = -np.exp(-0.5 * (y_pos - e2)**2)
            else:
                sum_coeff = 0.0
                kappa_left = kappa[j, i-1]
                kappa_interface = 2 * kappa_left * kappa[j, i] / (kappa_left + kappa[j, i] + 1e-12)
                coeff_left = kappa_interface / hx**2
                A[k, k-1] = -coeff_left
                sum_coeff += coeff_left
                
                kappa_right = kappa[j, i+1]
                kappa_interface = 2 * kappa[j, i] * kappa_right / (kappa[j, i] + kappa_right + 1e-12)
                coeff_right = kappa_interface / hx**2
                A[k, k+1] = -coeff_right
                sum_coeff += coeff_right
                
                if j > 0:
                    kappa_bottom = kappa[j-1, i]
                    kappa_interface = 2 * kappa_bottom * kappa[j, i] / (kappa_bottom + kappa[j, i] + 1e-12)
                    coeff_bottom = kappa_interface / hy**2
                    A[k, k - Nx] = -coeff_bottom
                    sum_coeff += coeff_bottom
                
                if j < Ny - 1:
                    kappa_top = kappa[j+1, i]
                    kappa_interface = 2 * kappa[j, i] * kappa_top / (kappa[j, i] + kappa_top + 1e-12)
                    coeff_top = kappa_interface / hy**2
                    A[k, k + Nx] = -coeff_top
                    sum_coeff += coeff_top
                A[k, k] = sum_coeff
    
    u = np.linalg.solve(A, b)
    u_grid = u.reshape((Ny, Nx))
    return u_grid