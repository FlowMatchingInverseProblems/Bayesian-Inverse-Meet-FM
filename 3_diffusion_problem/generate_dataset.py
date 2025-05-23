import numpy as np
from typing import List
from tqdm import tqdm
import os

from solver import solve_pde
from kl_extension import KLExpansion
from utils import get_d_from_u, create_mask

SIZE = 8192

savedir = f'data/several_points_gpt_{SIZE}'

os.makedirs(savedir, exist_ok=True)

kl = KLExpansion(grid=(64, 64))
kl.calculate_eigh()

d_arr = np.zeros((SIZE, 3), dtype='float')

for n_point in tqdm([3,4,5,6,7,8,9,10]):
    m_arr = np.zeros((SIZE, 16), dtype='float')
    e_arr = np.zeros((SIZE, 2), dtype='float')
    d_arr = np.zeros((SIZE, 3, n_point))
    for i in tqdm(range(SIZE)):
        m = np.random.normal(size = 16)
        m_arr[i] = m
        e = np.random.uniform(size = 2)
        e_arr[i] = e
        log_kappa = kl.expansion(m)
        u = solve_pde(np.exp(log_kappa), e[0], e[1])
        d = create_mask(u, n_point)
        d_arr[i] = d
    np.save(f'{savedir}/m_{n_point}', m_arr)
    np.save(f'{savedir}/e_{n_point}', e_arr)
    np.save(f'{savedir}/d_{n_point}', d_arr)

print('ok!')
