import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from dataclasses import dataclass
from tqdm import tqdm
import math
import wandb

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from natsort import natsorted

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from scipy.integrate import solve_ivp
from typing import List
from tqdm import tqdm
import yaml

N_EPOCH = 100_000

N = 7

LR = 8e-4

NAME = f"cfm_transformer_new_{N_EPOCH}_lr{int(LR*10_000)}e-4_no_causal"

MODEL_PATH = 'models'

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = torch.einsum("...,i->...i", t.float(), freqs)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

@dataclass
class GPTConfig:
    seq_length: int = 6  # Should be set to features_seq_len + 2 (e.g., 3 + 2 =5)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 32
    context_dim: int = 3

class CFMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projections
        self.feature_proj = nn.Linear(3, config.n_embd)
        self.x_t_proj = nn.Linear(6, config.n_embd)
        self.time_embed = TimestepEmbedder(config.n_embd)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.head = nn.Linear(config.n_embd, 6)

    def forward(self, t, x_t, features):
        features_permuted = features.permute(0, 2, 1)
        features_emb = self.feature_proj(features_permuted)
        
        x_t_emb = self.x_t_proj(x_t).unsqueeze(1)  # (B, 1, E)
        t_emb = self.time_embed(t).unsqueeze(1)     # (B, 1, E)
        
        sequence = torch.cat([t_emb, x_t_emb, features_emb], dim=1)
        
        x = sequence
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        return self.head(x[:, t_emb.shape[1]:t_emb.shape[1] + x_t_emb.shape[1], :])

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.qkv = nn.Linear(config.n_embd, 3*config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4*config.n_embd)
        self.fc2 = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

class SEIRDataset(Dataset):
    def __init__(self, path2files):
        self.features_list = natsorted(glob(path2files+'/features_*.npy'))[:N]
        self.targets_list = natsorted(glob(path2files+'/targets_*.npy'))[:N]
        self.features_chunks = [torch.tensor(np.load(x),dtype=torch.float32) for x in self.features_list]
        self.targets_chunks = [torch.tensor(np.load(x),dtype=torch.float32) for x in self.targets_list]
        self.num_chunks = len(self.features_list)
        self.total_samples = sum(chunk.shape[0] for chunk in self.features_chunks)
        self.idx = 0

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_id = idx // len(self.features_chunks[0])
        idx = idx % len(self.features_chunks[0])
        feature = self.features_chunks[chunk_id][idx]
        target = self.targets_chunks[chunk_id][idx]
        return feature, target
    
    def get_batch(self, idx = None):
        if idx is not None:
            self.idx = idx
        if self.idx >= self.num_chunks:
            self.idx = 0
        features_batch = self.features_chunks[self.idx]
        targets_batch = self.targets_chunks[self.idx]
        if idx is None:
            self.idx += 1
        return (features_batch, targets_batch)

class CFMTrainer:
    def __init__(self, config):
        self.config = config
        self.model = CFMTransformer(config)
        self.losses = [np.inf]
        self.val_losses = [np.inf]
        
        wandb.init(
            project="cfm-transformer_seir",
            config={
                "learning_rate": LR,
                "architecture": "CFMTransformer",
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd
            },
            name=f"cfm_transformer_new_{N_EPOCH}_lr5e4_no_causal_new_scheduler",
            entity="dp_new",
        )

    def train(self, dataset, val_dataset, epochs=100, batch_size=256):
        opt = optim.AdamW(self.model.parameters(), lr=LR)
        
        scheduler = CosineAnnealingLR(opt, T_max=50)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        def validate(model, val_dataset):
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i in range(len(val_dataset.features_chunks)):
                    features, targets = val_dataset.get_batch(idx=i)
                    features, targets = features.to(device), targets.to(device)
                    x0 = torch.randn_like(targets)
                    t = torch.rand(x0.shape[0]).type_as(x0)
                    xt = sample_conditional_pt(x0, targets, t, sigma=0.0)
                    ut = compute_conditional_vector_field(x0, targets)

                    vt = self.model(t, xt, features)
                    vt = vt[:, 0, :]
                    loss = torch.mean((vt - ut) ** 2)
                    total_val_loss += loss.item()
                    

            avg_val_loss = total_val_loss / len(val_dataset.features_chunks)
            return avg_val_loss
        
        def sample_conditional_pt(x0, x1, t, sigma):
            t = t.reshape(-1, *([1] * (x0.dim() - 1)))
            mu_t = t * x1 + (1 - t) * x0
            epsilon = torch.randn_like(x0)
            return mu_t + sigma * epsilon
        
        def compute_conditional_vector_field(x0, x1):
            return x1 - x0
        
        for epoch in tqdm(range(epochs)):
            self.model.train()
            total_loss = 0
            
            for i in tqdm(range(len(dataset.features_chunks)), desc=f"Epoch {epoch+1}"):
                features, targets = dataset.get_batch(idx=i)
                features, targets = features.to(device), targets.to(device)
                
                x0 = torch.randn_like(targets)
                t = torch.rand(x0.shape[0]).type_as(x0)
                xt = sample_conditional_pt(x0, targets, t, sigma=0.0)
                ut = compute_conditional_vector_field(x0, targets)

                vt = self.model(t, xt, features)
                vt = vt[:, 0, :]
                loss = torch.mean((vt - ut) ** 2)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                
                wandb.log({"batch_loss": loss.item()})
            
            avg_loss = total_loss / len(dataset.features_chunks)
            
            if val_dataset is not None:
                avg_val_loss = validate(self.model, val_dataset)
                scheduler.step()
                wandb.log({
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_loss,
                    "avg_val_loss": avg_val_loss,
                })
                output_str = f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val_loss : {avg_val_loss:.4f}"
                if avg_val_loss < min(self.val_losses):
                    torch.save(self.model.state_dict(), f"{MODEL_PATH}/{NAME}_best.pth")
                    wandb.save(f"{MODEL_PATH}/{NAME}_best.pth")
                self.val_losses.append(avg_val_loss)
            else:
                scheduler.step()
                wandb.log({
                    "epoch": epoch + 1,
                    "avg_train_loss": avg_loss,
                })
                output_str = f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
                if avg_loss < min(self.losses):
                    torch.save(self.model.state_dict(), f"{MODEL_PATH}/{NAME}_best.pth")
                    wandb.save(f"{MODEL_PATH}/{NAME}_best.pth")
            
            self.losses.append(avg_loss)
            print(output_str)
    
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



if __name__ == "__main__":
    config = GPTConfig()
    dataset = SEIRDataset(path2files='seir_data_cust_points_8192')
    val_dataset = SEIRDataset(path2files='seir_data_cust_points_512')
    trainer = CFMTrainer(config)
    trainer.train(dataset, val_dataset, epochs=N_EPOCH, batch_size=8192)
    plt.figure(dpi=300)
    plt.plot(trainer.losses)
    plt.yscale('log')
    plt.savefig('losses.png')
    wandb.log({"loss_plot": wandb.Image('losses.png')})
    torch.save(trainer.model.state_dict(), f"{NAME}.pth")
    trainer.model.load_state_dict(torch.load(f"{MODEL_PATH}/{NAME}_best.pth", map_location='cpu'))
    
    device = torch.device('cpu')
    trainer.model.to(device)
    m = np.array([0.4, 0.3, 0.3, 0.1, 0.15, 0.6])
    ref_sol = solve_SEIR(m)

    N_iter = 50

    error_sol = {0: ''}

    parameters = {0: []}

    for N in [2,3,4,5,6,7,8,9]:
        e = np.linspace(1,3,N)
        d = d_by_m_e(m,e)
        feature = np.vstack([d,e])
        feature_torch = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        

        def ode_func(t, x, feat):
            x = x.reshape(1,-1)
            t = torch.tensor([t], dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            output = trainer.model(t,x, feat)
            return output.detach().cpu()[0][0]

        error_sol_i, m_arr = [], []
        means, stds = [], []
        parameters[N] = []
        for _ in tqdm(range(N_iter)):
            m0 = np.random.uniform(size=6)
            solution = solve_ivp(ode_func, t_span=[0, 1], y0=m0, t_eval=None, args=(torch.tensor(feature, dtype=torch.float32).unsqueeze(0),))
            m_pred = solution.y[:,-1]
            m_arr.append(m_pred)

            d_pred = d_by_m_e(m_pred,e)

            sol_pred = solve_SEIR(m_pred)
            parameters[N].append(m_pred)
            if sol_pred.shape != ref_sol.shape:
                print(sol_pred.shape)
                print(ref_sol.shape)
                pass
            else:
                error_sol_i.append(np.linalg.norm(ref_sol - sol_pred)/np.linalg.norm(ref_sol))

        error_sol[N] = f'{np.mean(error_sol_i)*100 : .3f}%+-{np.std(error_sol_i)*100 :.3f}%'
        means.append(np.mean(error_sol_i))
        stds.append(np.std(error_sol_i))


    del error_sol[0]


    from pprint import pprint

    pprint(error_sol)
    
    with open(f'logs/{NAME}_output.yaml', 'w') as f:
        yaml.dump(error_sol, f)

    wandb.finish()
