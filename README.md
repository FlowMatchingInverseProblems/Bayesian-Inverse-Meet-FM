# Bayesian Inverse Problems Meet Flow Matching

## Structure

### `1_simple_non_linear`

* `evaluate.py`: Evaluation of model performance.
* `mcmc.py`: Implementation of Markov Chain Monte Carlo sampling.
* `train.py`: Training script for the Flow Matching model.
* `models/`: Neural network architectures used in the problem.

### `2_seir_problem`

A Bayesian treatment of the SEIR (Susceptible-Exposed-Infectious-Removed) epidemiological model.

* `generate_dataset.py`: Script for simulating data from the SEIR model.
* `mcmc.py`: MCMC sampling for posterior inference.
* `train.py`: Training the inference network.
* `logs/`: Directory for storing logs and results.
* `models/`: Contains model definitions.

### `3_diffusion_problem`

Solving a diffusion PDE with uncertainty in input parameters.

* `generate_dataset.py`: Dataset generation via PDE simulation.
* `kl_expansion.py`: Karhunen–Loève expansion for dimensionality reduction.
* `mcmc.py`: Posterior sampling using MCMC.
* `solver.py`: Numerical solver for the diffusion equation.
* `train.py`: Model training script.
* `utils.py`: Utility functions for preprocessing, etc.
* `logs/`, `models/`: Output directories for logs and model checkpoints.

## Requirements

To run the scripts, install the required Python packages.

- For simple non-linear model

```bash
pip install numpy scipy matplotlib tqdm jax flax optax 
```
- For others


```bash
pip install numpy scipy matplotlib torch wandb
```

## Usage

Each problem directory can be executed independently. Typically, you would follow these steps:

1. Generate data using `generate_dataset.py` (if available).
2. Train a model using `train.py`.
3. Run inference using `mcmc.py`. (optiobal)
4. Evaluate results using `evaluate.py` (if available).


Also you can find some data samples [here](https://disk.yandex.ru/d/Tx3QTcNW5ILOVQ)