import numpy as np
import matplotlib.pyplot as plt



sigma_noise = 0.0001 

def forward_model(m, e):
    return e**2 * m**3 + m * np.exp(-abs(0.2 - e)) + np.random.normal(0, scale=1e-4)

def likelihood(d, e, m, sigma):
    model_output = forward_model(m, e)
    return np.exp(-0.5 * ((d - model_output) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

def prior(m):
    return 1.0 if 0 <= m <= 1 else 0.0

def posterior(d, e, m, sigma):
    return likelihood(d, e, m, sigma) * prior(m)

def mcmc(d, e, sigma, n_samples=10000, proposal_std=0.1):
    samples = []
    current_m = np.random.uniform(0, 1)
    for _ in range(n_samples):
        proposed_m = current_m + np.random.normal(0, proposal_std)
    
        p_current = posterior(d, e, current_m, sigma)
        p_proposed = posterior(d, e, proposed_m, sigma)
        acceptance_ratio = p_proposed / p_current if p_current > 0 else 0
        
        if np.random.rand() < acceptance_ratio:
            current_m = proposed_m
        samples.append(current_m)

    return np.array(samples)

m = 0.8
e = 0.2
d = forward_model(m,e)
d_observed = forward_model(m,e) 
e_observed = e  


n_samples = 10_000
samples = mcmc(d=d_observed, e=e_observed, sigma=sigma_noise, n_samples=n_samples, proposal_std=0.1)

plt.figure(figsize=(8,6))
plt.hist(samples, bins=50, density=True, alpha=0.7, label="Posterior")
plt.xlabel("m")
plt.ylabel("Density")
plt.title(fr"MCMC with n_samples {n_samples}, $\rho (m \vert e = {e}, d = {d : .2f})$, true $m$ = {m}")
plt.legend()
plt.xlim(0,1)
plt.savefig('mcmc.png')
plt.show()
