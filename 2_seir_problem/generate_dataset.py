import numpy as np
from scipy.integrate import solve_ivp
from typing import List
from tqdm import tqdm
import os

SIZE = 8192

datadir = f'seir_data_cust_points_{SIZE}'
os.makedirs(datadir, exist_ok=True)

def d_by_m_e(m: List[int], e: List[int]):
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

    solution = solve_ivp(seir_model, t_span=[0,4], y0=y0, t_eval=e, args=(beta, alpha, gamma))
    return solution.y[2:]

def generate_e(n: int):
    boundaries = np.linspace(1, 3, n + 1)
    
    e = np.zeros(n)
    for i in range(n):
        low = boundaries[i]
        high = boundaries[i + 1]
        e[i] = np.random.uniform(low, high)
    return e

n_numbers = [4, 5, 6, 7, 8, 9, 10]

for n in tqdm(n_numbers):
    m_arr = np.zeros((SIZE, 6), dtype='float')
    features_arr = np.zeros((SIZE, 3, n), dtype='float')
    features_noise_arr = np.zeros((SIZE, 3, n), dtype='float')
    for i in tqdm(range(SIZE)):
        constraint = True
        while constraint:
            m = np.random.uniform(size=6)
            e = generate_e(n)
            d = d_by_m_e(m,e)
        noise = np.vstack([np.random.normal(0, 2, size=n), np.random.normal(0, 1, size=n)])
        features = np.vstack([d,e])
        features_noise = np.vstack([d + noise,e])
        m_arr[i] = m
        features_arr[i] = features
        features_noise_arr[i] = features_noise
    np.save(f'{datadir}/targets_{n}.npy', m_arr)
    np.save(f'{datadir}/features_{n}.npy', features_arr)
    np.save(f'{datadir}/features_noise_{n}.npy', features_noise_arr)
