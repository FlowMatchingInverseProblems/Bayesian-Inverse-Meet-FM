import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from typing import List

# Define constants
sigma_noise = 0.09
N_values = [2, 3, 4, 5, 6, 7, 8] 
N_samples = [5_000, 10_000]


def solve_SEIR(m: List[int], t = np.linspace(0,4,50)):
    beta1, alpha, gamma_r, gamma_d1, beta2, gamma_d2 = m
    tau = 2.1
    
    def beta(t):
        return beta1 + 0.5 * np.tanh(7 * (t - tau)) * (beta2 - beta1)

    def gamma_d(t):
        return gamma_d1 + 0.5 * np.tanh(7 * (t - tau)) * (gamma_d2 - gamma_d1)

    def gamma(t):
        return gamma_r + gamma_d(t)

    def seir_model(t, y, beta, alpha, gamma):
        S, E, I, R = y
        dSdt = -beta(t) * S * I
        dEdt = beta(t) * S * I - alpha * E
        dIdt = alpha * E - gamma(t) * I
        dRdt = (gamma_r + gamma_d(t)) * I
        return [dSdt, dEdt, dIdt, dRdt]

    S0, E0, I0, R0 = 99, 1, 0, 0
    y0 = [S0, E0, I0, R0]

    solution = solve_ivp(seir_model, t_span=[0,4], y0=y0, t_eval=t, args=(beta, alpha, gamma))
    return solution.y

def d_by_m_e(m: List[float], e: List[float]):
    beta1, alpha, gamma_r, gamma_d1, beta2, gamma_d2 = m
    tau = 2.1
    
    def beta(t):
        return beta1 + 0.5 * np.tanh(7 * (t - tau)) * (beta2 - beta1)
    
    def gamma_d(t):
        return gamma_d1 + 0.5 * np.tanh(7 * (t - tau)) * (gamma_d2 - gamma_d1)
    
    def gamma(t):
        return gamma_r + gamma_d(t)
    
    def seir_model(t, y, beta, alpha, gamma):
        S, E, I, R = y
        dSdt = -beta(t) * S * I
        dEdt = beta(t) * S * I - alpha * E
        dIdt = alpha * E - gamma(t) * I
        dRdt = gamma(t) * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    S0, E0, I0, R0 = 99, 1, 0, 0
    y0 = [S0, E0, I0, R0]
    solution = solve_ivp(seir_model, [0, 4], y0, t_eval=e, args=(beta, alpha, gamma))
    return solution.y[2:]

def forward_model(m, e):
    sol = d_by_m_e(m, e)
    return sol.flatten()

def log_likelihood(m, e, d_obs):
    d_pred = forward_model(m, e)
    if d_pred.shape[0] != d_obs.shape[0]:
        d_obs = d_obs[:d_pred.shape[0]]
    residual = d_obs - d_pred
    return -0.5 * np.sum((residual / sigma_noise) ** 2)

def log_prior(m):
    return 0.0 if np.all((m >= 0) & (m <= 1)) else -np.inf

def log_posterior(m, e, d_obs):
    lp = log_prior(m)
    return lp + log_likelihood(m, e, d_obs) if lp != -np.inf else -np.inf

def metropolis_hastings(n_samples, initial_m, proposal_std, e, d_obs):
    samples = np.zeros((n_samples, len(initial_m)))
    m_current = initial_m.copy()
    log_post_current = log_posterior(m_current, e, d_obs)
    accepted = 0
    for i in tqdm(range(n_samples)):
        m_proposal = m_current + np.random.normal(0, proposal_std, size=len(m_current))
        log_post_proposal = log_posterior(m_proposal, e, d_obs)
        if np.log(np.random.rand()) < log_post_proposal - log_post_current:
            m_current = m_proposal
            log_post_current = log_post_proposal
            accepted += 1
        samples[i, :] = m_current
    return samples, accepted / n_samples

from pprint import pprint

N = 5
m_true = np.array([0.4, 0.3, 0.3, 0.1, 0.15, 0.6])
e_observed = np.linspace(1, 3, N)
d_true = forward_model(m_true, e_observed)
sol_ref = solve_SEIR(m_true)
print(sol_ref.shape)

relative_errors = {}
for N in N_values:
    e_observed = np.linspace(1, 3, N)
    d_true = forward_model(m_true, e_observed)
    rel_errors_N = []
    for n_samples in N_samples:
        initial_m = np.random.rand(6)
        samples, _ = metropolis_hastings(n_samples, initial_m, 0.02, e_observed, d_true)
        sol_pred = solve_SEIR(np.mean(samples, axis=0))
        rel_error = np.linalg.norm(sol_ref - sol_pred) / np.linalg.norm(sol_ref)
        rel_errors_N.append(rel_error)
    relative_errors[N] = rel_errors_N
    print(N,':\n')
    pprint(relative_errors[N])
for N, errors in relative_errors.items():
    pprint(f"N = {N}: {errors}")

import pickle

with open('seir_rel_errors_mcmc_new.pkl', 'wb') as f:
    pickle.dump(relative_errors,f)