import os
import time
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax import linen as nn
from flax.training import train_state

import pickle
from scipy.integrate import solve_ivp


savedir = "models/simple_64_another_2"

class MLP(nn.Module):
    dim: int
    out_dim: int = 1
    w: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
model = MLP(dim=4)

@jax.jit
def predict(params, inputs):
    return model.apply({"params": params}, inputs)

with open(f'{savedir}/w.pkl', 'rb') as f:
    params = pickle.load(f)

def ode_function(t, m, d, e):
    inputs = jnp.array([m[0], d, e, t]).reshape(1,-1)
    vt = predict(params, inputs)
    return vt[0]



def ode_function(t, m, d, e):
    inputs = jnp.array([m[0], d, e, t])
    return predict(params, inputs)[0]

def d_by_m_e(m, e):
    noise = np.random.normal(scale=1e-4, size=1).item()
    d = np.power(e, 2) * np.power(m, 3) + m * np.exp(-np.abs(0.2 - e)) + noise
    return d

errors, m_sol, d_err = [], [], []

for _ in tqdm(range(1000)):
    m0 = np.random.uniform(size=1).item()
    m = 0.2
    e = 0.1 
    d = d_by_m_e(m, e)

    solution = solve_ivp(ode_function, t_span=[0, 1], y0=[m0], t_eval=None, args=(d, e))
    m_sol.append(solution.y[0][-1])
    errors.append(np.abs(m_sol[-1] - m))
    d_err.append(np.abs(d_by_m_e(m_sol[-1], e) - d))

print(np.mean(errors))