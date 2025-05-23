import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_pde(kappa, e1=0.5, e2=0.5):
    Ny, Nx = kappa.shape               
    hx = hy = 1.0 / (Nx - 1)            
    N  = Ny * Nx                      

    A = sp.lil_matrix((N, N))
    b = np.zeros(N)

    for j in range(Ny):
        y = j * hy
        for i in range(Nx):
            k = i + j * Nx             

            if i == 0:
                A[k, k] = 1.0
                b[k]    =  np.exp(-0.5 * (y - e1)**2)
                continue
            if i == Nx - 1:
                A[k, k] = 1.0
                b[k]    = -np.exp(-0.5 * (y - e2)**2)
                continue

            if j == 0:
                A[k, k]       =  1.0
                A[k, k + Nx]  = -1.0
                continue
            if j == Ny - 1:
                A[k, k]       =  1.0
                A[k, k - Nx]  = -1.0
                continue
            a_w = 0.5 * (kappa[j, i]     + kappa[j, i-1])
            a_e = 0.5 * (kappa[j, i]     + kappa[j, i+1])
            a_s = 0.5 * (kappa[j, i]     + kappa[j-1, i])
            a_n = 0.5 * (kappa[j, i]     + kappa[j+1, i])

            diag = (a_w + a_e) / hx**2 + (a_s + a_n) / hy**2

            A[k, k]         =  diag
            A[k, k - 1]     = -a_w / hx**2        
            A[k, k + 1]     = -a_e / hx**2        
            A[k, k - Nx]    = -a_s / hy**2       
            A[k, k + Nx]    = -a_n / hy**2       

    u = spla.spsolve(A.tocsr(), b)
    return u.reshape((Ny, Nx))


