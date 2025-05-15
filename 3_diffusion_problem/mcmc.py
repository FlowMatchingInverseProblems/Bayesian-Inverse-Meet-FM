import numpy as np
from tqdm import tqdm
import os

from solver import solve_pde
from kl_extension import KLExpansion
from utils import create_mask

sigma_noise = 0.01
N_values = range(2, 8) 
N_samples = [5000, 10_000]
proposal_std=0.01

kl = KLExpansion(grid=(64, 64))
kl.calculate_eigh()

def forward_model(m, e, n_points):
    log_kappa = kl.expansion(m)
    u = solve_pde(np.exp(log_kappa), e[0], e[1])
    return create_mask(u, n_points)[0]

def pde_sol(m, e):
    log_kappa = kl.expansion(m)
    u = solve_pde(np.exp(log_kappa), e[0], e[1])
    return u

def log_likelihood(m, e, d_obs, n_points):
    d_pred = forward_model(m, e, n_points)
    residual = d_obs - d_pred
    return -0.5 * np.sum((residual / sigma_noise) ** 2)

def log_prior(m):
    mean = 0.0
    std_dev = 1.0
    return -0.5 * np.sum((m / std_dev) ** 2)

def log_posterior(m, e, d_obs, n_points):
    lp = log_prior(m)
    return lp + log_likelihood(m, e, d_obs, n_points) if lp != -np.inf else -np.inf

def metropolis_hastings(n_samples, initial_m, proposal_std, e, d_obs, n_points):
    samples = np.zeros((n_samples, len(initial_m)))
    m_current = initial_m.copy()
    log_post_current = log_posterior(m_current, e, d_obs, n_points)
    accepted = 0
    for i in tqdm(range(n_samples)):
        m_proposal = m_current + np.random.normal(0, proposal_std, size=len(m_current))
        log_post_proposal = log_posterior(m_proposal, e, d_obs, n_points)
        if np.log(np.random.rand()) < log_post_proposal - log_post_current:
            m_current = m_proposal
            log_post_current = log_post_proposal
            accepted += 1
        samples[i, :] = m_current
    return samples, accepted / n_samples

e_observed = [0.9, 0.2]
m_true = np.array([-1.80560093,  0.03123921,  0.0365751 , -1.12853305,  0.30499035,
       -0.80850724,  0.24399505, -1.06747149,  0.67463376,  1.60872746,
        0.66880287,  0.00241754,  0.77226669,  0.22825061, -1.47802239,
        0.31206404])
sol_ref = pde_sol(m_true, e_observed)

from pprint import pprint

relative_errors = {}
for N in N_values:
    d_true = forward_model(m_true, e_observed, N)
    rel_errors_N = []
    for n_samples in N_samples:
        initial_m = np.random.normal(size=16)
        samples, _ = metropolis_hastings(n_samples, initial_m, proposal_std, e_observed, d_true, N)
        sol_pred = pde_sol(np.mean(samples, axis=0), e_observed)
        rel_error = np.linalg.norm(sol_ref - sol_pred) / np.linalg.norm(sol_ref)
        rel_errors_N.append(rel_error)
    relative_errors[N] = rel_errors_N
    print(N,':\n')
    pprint(relative_errors[N])
    

for N, errors in relative_errors.items():
    print(f"N = {N}: {errors}")
    
import pickle

with open('diff_errors.pkl', 'wb') as f:
    pickle.dump(relative_errors,f)
