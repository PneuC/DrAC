import math
import numpy as np
import torch
from torch import nn
from rl.nets.auxiliary import LearnableCoeffient
from rl.nets.base import *


class ReparamGaussPolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, hidden_layers=2, activation='relu', layer_norm=False):
        super().__init__()
        self.act_dim = act_dim
        self.main = create_unihid_mlp(obs_dim, act_dim * 2, hidden_size, hidden_layers, activation, 'none', layer_norm=layer_norm)

    def forward(self, obs, deterministic=False):
        mu, log_std = self.main(obs).split(self.act_dim, -1)
        if deterministic:
            return torch.tanh(mu), None
        return squashed_gauss_rsample(mu, log_std)


class MLPLatentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, hidden_layers=2, z_dim=None, activation='relu', layer_norm=False):
        super().__init__()
        if z_dim is None:
            z_dim = max(act_dim, 16)
        self.act_dim = act_dim
        self.z_dim = z_dim
        self.main = create_unihid_mlp(obs_dim + z_dim, act_dim, hidden_size, hidden_layers, activation, 'tanh', layer_norm=layer_norm)

    def forward(self, obs):
        z = torch.rand(*obs.shape[:-1], self.z_dim, device=obs.device)
        x = torch.cat([obs, z], -1)
        return self.main(x)

    def rept_sample(self, obs, rept):
        obs_rept = obs.expand(rept, *obs.shape)
        z = torch.rand(rept, *obs.shape[:-1], self.z_dim, device=obs.device)
        x = torch.cat([obs_rept, z], -1)
        return self.main(x)


class MLPDiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, hidden_layers=2, activation='relu', layer_norm=False, time_emb_dim=16, steps=20):
        super().__init__()
        self.act_dim = act_dim
        self.main = create_unihid_mlp(obs_dim + act_dim + time_emb_dim, act_dim, hidden_size, hidden_layers, activation, 'none', layer_norm=layer_norm)
        self.time_embedder = create_unihid_mlp(time_emb_dim, time_emb_dim, time_emb_dim * 2, 1, activation, 'none', layer_norm=layer_norm)
        self.T = steps
        self.time_emd_dim = time_emb_dim
        self.schedule = BetaScheduleCoefficients.vp_from_T(self.T)

    def forward(self, obs):
        device = obs.device
        outer_shape = None if obs.ndim == 1 else obs.shape[:-1]
        times = torch.arange(self.T - 1, -1, -1, device=device)
        time_embs = self.scaled_sinusoidal_encoding(times, self.time_emd_dim, outer_shape)

        x_shape = (self.act_dim,) if obs.ndim == 1 else (*outer_shape, self.act_dim)
        x = torch.randn(*x_shape, device=device)
        for i, te in enumerate(time_embs):
            mu = self.main(torch.cat([obs, x, te], dim=-1))
            wx = self.schedule.sqrt_recip_alphas_cumprod[-i]
            wz = self.schedule.sqrt_recipm1_alphas_cumprod[-i]
            x_recon = torch.clip(wx * x - wz * mu, -1, 1)
            w_recon = self.schedule.posterior_mean_coef1[-i]
            wx = self.schedule.posterior_mean_coef2[-i]
            mean = w_recon * x_recon + wx * x
            std = math.exp(self.schedule.posterior_log_variance_clipped[-i] / 2)
            x = mean + float(i > 0) * std * torch.randn_like(x)
        return x

    def rept_sample(self, obs, rept):
        return self.forward(obs.expand(rept, *obs.shape))

    def scaled_sinusoidal_encoding(self, t: torch.Tensor, theta: int = 10000, batch_shape=None):
        assert self.time_emd_dim % 2 == 0, "dim must be even"
        device = t.device
        dtype = t.dtype
        scale = 1 / (self.time_emd_dim ** 0.5)
        half_dim = self.time_emd_dim // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=dtype) / half_dim
        inv_freq = theta ** -freq_seq
        emb = torch.einsum('..., j -> ...j', t, inv_freq)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb * scale
        if batch_shape is not None:
            target_shape = (*batch_shape, self.T, self.time_emd_dim)
            emb = emb.broadcast_to(target_shape)
            emb = self.time_embedder(emb)
        return [emb[..., i, :] for i in range(self.T)]


if __name__ == '__main__':
    md = MLPDiffusionPolicy(10, 10)
    print(md.forward(torch.rand(10)).shape)
    print(md.forward(torch.rand(5, 10)).shape)
    pass
