import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from typing import Callable, Tuple

def kernel(x, z, sigma_v: float=1.0, l_2: float =0.1):
    dist = cdist(x, z)
    return sigma_v**2 * np.exp(-dist**2 / (2 * l_2))


class KLExpansion():
    def __init__(self, n_m: int = 16, 
                 sigma_v: float = 1,
                 l_2: float = 0.1,
                 kernel = kernel,
                 grid: Tuple[int] = (64,64)):
        self.n_m = n_m
        self.sigma_v = sigma_v
        self.l_2 = l_2,
        self.kernel = kernel
        grid_x, grid_y = grid
        self.grid_x = grid_x
        self.grid_y = grid_y

    def calculate_eigh(self):
        x = np.linspace(0, 1, self.grid_x)
        y = np.linspace(0, 1, self.grid_y)
        X, Y = np.meshgrid(x, y)
        self.X_mesh = X
        self.Y_mesh = Y
        mesh = np.column_stack((X.flatten(), Y.flatten()))
        self.mesh = mesh
        K = kernel(mesh, mesh)
        eigenvalues, eigenvectors = eigh(K)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues_trunc = eigenvalues[:self.n_m]
        eigenvectors_trunc = eigenvectors[:, :self.n_m]
        self.eigenvalues = eigenvalues_trunc
        self.eigenvectors = eigenvectors_trunc

    def expansion(self, m_i):
        return np.sum(m_i[:, None] * np.sqrt(self.eigenvalues[:, None]) * self.eigenvectors.T, axis=0).reshape(self.grid_x, self.grid_y)    
    
    def visualize_field(self, field, title=None):
        plt.figure(figsize=(12, 4), dpi=200)
        plt.subplot(121)
        plt.contourf(self.X_mesh, self.Y_mesh, field, levels=50, cmap='viridis')
        plt.colorbar()
        if title is not None:
            plt.title(title)
        plt.show()




