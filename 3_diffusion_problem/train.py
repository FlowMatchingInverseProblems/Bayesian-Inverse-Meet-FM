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

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from natsort import natsorted


import os

from pprint import pprint

from solver import solve_pde
from kl_extension import KLExpansion
from utils import get_d_from_u, create_mask, create_mask_

from scipy.integrate import solve_ivp
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

N_EPOCH = 100_000

N = 8
N_samples = 4096

N_iter = 50

LR = 8e-4

NAME = f"cfm_transformer_new_{N_EPOCH}_lr{int(LR*10_000)}e-4_no_causal_new_eval_uniform"

MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=32):
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

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
#         y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=torch.ones((T,T),dtype=q.dtype,device=q.device))
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

@dataclass
class GPTConfig:
    n_layer: int = 8
    n_head: int = 4
    n_embd: int = 64

class CFMTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projections
        self.d_feature_proj = nn.Linear(3, config.n_embd)
        self.e_feature_proj = nn.Linear(2, config.n_embd)
        self.x_t_proj = nn.Linear(16, config.n_embd)
        self.time_embed = TimestepEmbedder(config.n_embd)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.head = nn.Linear(config.n_embd, 16)

    def forward(self, t, x_t, e_features, d_features):
        
        d_features_permuted = d_features.permute(0, 2, 1)
        d_features_emb = self.d_feature_proj(d_features_permuted)
        
        e_features_emb = self.e_feature_proj(e_features).unsqueeze(1) # (B, 1, E)
        x_t_emb = self.x_t_proj(x_t).unsqueeze(1)  # (B, 1, E)
        t_emb = self.time_embed(t).unsqueeze(1)     # (B, 1, E)
        
        sequence = torch.cat([t_emb, x_t_emb, e_features_emb, d_features_emb], dim=1)
        
        x = sequence
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        return self.head(x[:, t_emb.shape[1]:t_emb.shape[1] + x_t_emb.shape[1], :])

class DiffDataset(Dataset):
    def __init__(self, path2files):
        self.d_features_list = natsorted(glob(path2files+'/d_*.npy'))[:N]
        self.e_features_list = natsorted(glob(path2files+'/e_*.npy'))[:N]
        self.targets_list = natsorted(glob(path2files+'/m_*.npy'))[:N]
        self.d_features_chunks = [torch.tensor(np.load(x),dtype=torch.float32) for x in self.d_features_list]
        self.e_features_chunks = [torch.tensor(np.load(x),dtype=torch.float32) for x in self.e_features_list]
        self.targets_chunks = [torch.tensor(np.load(x), dtype=torch.float32) for x in self.targets_list]
        self.num_chunks = len(self.d_features_list)
        self.total_samples = sum(chunk.shape[0] for chunk in self.d_features_chunks)
        self.idx = 0

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_id = idx // len(self.d_features_chunks[0])
        idx = idx % len(self.d_features_chunks[0])
        d_feature = self.d_features_chunks[chunk_id][idx]
        e_feature = self.e_features_chunks[chunk_id][idx]
        target = self.targets_chunks[chunk_id][idx]
        return d_feature, e_feature, target
    
    def get_batch(self, idx = None):
        if idx is not None:
            self.idx = idx
        if self.idx >= self.num_chunks:
            self.idx = 0
        d_features_batch = self.d_features_chunks[self.idx]
        e_features_batch = self.e_features_chunks[self.idx]
        targets_batch = self.targets_chunks[self.idx]
        if idx is None:
            self.idx += 1
        return d_features_batch, e_features_batch, targets_batch

class CFMTrainer:
    def __init__(self, config):
        self.config = config
        self.model = CFMTransformer(config)
        self.losses = [np.inf]
        self.val_losses = [np.inf]
        
        # Initialize wandb
        wandb.init(
            project="cfm-transformer_diffusion",
            config={
                "learning_rate": LR,
                "architecture": "CFMTransformer",
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd
            },
            name=NAME,
            entity="dp_new",
        )

    def train(self, dataset, val_dataset = None, epochs=100, batch_size=256):
        
        opt = optim.AdamW(self.model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, eta_min=1e-6)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        def validate(model, val_dataset):
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i in range(len(val_dataset.d_features_chunks)):
                    d_features, e_features, targets = val_dataset.get_batch(idx=i)
                    d_features, e_features, targets = d_features.to(device), e_features.to(device), targets.to(device)

                    x0 = torch.rand_like(targets)
                    t = torch.rand(x0.shape[0]).type_as(x0)
                    xt = sample_conditional_pt(x0, targets, t, sigma=0.0)
                    ut = compute_conditional_vector_field(x0, targets)

                    vt = self.model(t, xt, e_features, d_features)
                    vt = vt[:,0,:]
                    loss = torch.mean((vt - ut) ** 2)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataset.d_features_chunks)
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
            
            for i in tqdm(range(len(dataset.d_features_chunks)), desc=f"Epoch {epoch+1}"):
                d_features, e_features, targets = dataset.get_batch(idx=i)
                d_features, e_features, targets = d_features.to(device), e_features.to(device), targets.to(device)
                
                x0 = torch.rand_like(targets)
                t = torch.rand(x0.shape[0]).type_as(x0)
                xt = sample_conditional_pt(x0, targets, t, sigma=0.0)
                ut = compute_conditional_vector_field(x0, targets)

                vt = self.model(t, xt, e_features, d_features)
                vt = vt[:,0,:]
                loss = torch.mean((vt - ut) ** 2)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
                wandb.log({"batch_loss": loss.item()})
            
            avg_loss = total_loss / len(dataset.d_features_list)
            scheduler.step()
        
            
            if val_dataset is not None:
                avg_val_loss = validate(self.model, val_dataset)
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

from kl_extension import KLExpansion
from utils import get_d_from_u


def create_mask_(array, rows, cols, cell_size, points_count):
    result = np.zeros((3, points_count))
    used_sectors = set()
    for k in range(points_count):
        while True:
            i = np.random.randint(rows)
            j = np.random.randint(cols)
            if (i, j) not in used_sectors:
                used_sectors.add((i, j))
                break

        start_row = i * cell_size
        end_row = (i + 1) * cell_size
        start_col = j * cell_size
        end_col = (j + 1) * cell_size

        center_row = start_row + cell_size // 2 + np.random.randint(-cell_size // 7, cell_size // 7)
        center_col = start_col + cell_size // 2 + np.random.randint(-cell_size // 7, cell_size // 7)

        result[0, k] = array[center_row, center_col]
        result[1, k] = center_row / (array.shape[0] - 1)
        result[2, k] = center_col / (array.shape[1] - 1)
    return result

def create_mask(arr, n_points):
    div = 3 + n_points // 10
    cell_size = 64 // div
    rows = div
    cols = div
    mask = create_mask_(arr, rows, cols, cell_size, n_points)
    return mask


if __name__ == "__main__":
    config = GPTConfig()
    dataset = DiffDataset(path2files='data/several_points_gpt_4096')
    val_dataset = DiffDataset(path2files='data/several_points_gpt_256')
    trainer = CFMTrainer(config)
    trainer.train(dataset, val_dataset, epochs=N_EPOCH, batch_size=8192)
    plt.figure(dpi=300)
    plt.plot(trainer.losses)
    plt.yscale('log')
    plt.savefig('losses.png')
    wandb.log({"loss_plot": wandb.Image('losses.png')})
    torch.save(trainer.model.state_dict(), f"{MODEL_PATH}/{NAME}.pth")
    trainer.model.load_state_dict(torch.load(f"{MODEL_PATH}/{NAME}_best.pth"))
    m = np.array([-1.80560093,  0.03123921,  0.0365751 , -1.12853305,  0.30499035,
       -0.80850724,  0.24399505, -1.06747149,  0.67463376,  1.60872746,
        0.66880287,  0.00241754,  0.77226669,  0.22825061, -1.47802239,
        0.31206404])


    n_points = 7
    div = 3 + n_points // 10
    cell_size = 64 // div
    rows = div
    cols = div

    mask_arr = create_mask_(np.ones((64,64)),rows, cols, cell_size, n_points)
    kl = KLExpansion(grid=(64, 64))
    kl.calculate_eigh()

    log_kappa_ref = kl.expansion(m)
    kappa_ref = np.exp(log_kappa_ref)
    e_ref = [0.9, 0.1]
    u_ref = solve_pde(log_kappa_ref, e_ref[0], e_ref[1])

    device = torch.device('cpu')
    trainer.model.to(device)

    error_sol = {0: ''}
    N_iter = 50

    for N in tqdm([2,3,4,5,6,7,8,9]):
        print(N, ':')
        d_ref = create_mask(u_ref, N)
        d_feature_torch = torch.tensor(d_ref, dtype=torch.float32).unsqueeze(0)
        e_feature_torch = torch.tensor([e_ref], dtype=torch.float32)

        def ode_func(t, x, d_feat, e_feat):
            x = x.reshape(1,-1)
            t = torch.tensor([t], dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            output = trainer.model(t,x, e_feat, d_feat)
            return output.detach().cpu()[0][0]

        error_sol_i, m_arr = [], []
        means, stds = [], []
        for _ in tqdm(range(N_iter)):
            m0 = np.random.uniform(size=16)
            solution = solve_ivp(ode_func, t_span=[0, 1], y0=m0, t_eval=None, args=(d_feature_torch,e_feature_torch))
            m_pred = solution.y[:,-1]
            m_arr.append(m_pred)
            log_kappa_pred = kl.expansion(m_pred)
            kappa_pred = np.exp(log_kappa_pred)
            sol_pred = solve_pde(kappa_pred, e_ref[0],e_ref[1])
            error_sol_i.append(np.linalg.norm(u_ref - sol_pred)/np.linalg.norm(u_ref))
            d_pred = create_mask(sol_pred, N)
        
        N = str(N)
        error_sol[N] = f'{np.mean(error_sol_i)*100 : .3f}%+-{np.std(error_sol_i)*100 :.3f}%'
        means.append(np.mean(error_sol_i))
        stds.append(np.std(error_sol_i))


    del error_sol[0]
    from pprint import pprint

    pprint(error_sol)

    import json
    with open("logging/results.json", "w") as f:
        json.dump(error_sol, f)

    artifact = wandb.Artifact("results", type="dataset")
    artifact.add_file("logging/results.json")
    wandb.log_artifact(artifact)

    wandb.finish()
