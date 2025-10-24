from dataclasses import dataclass
from typing import Tuple, Protocol

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


def create_activation(key):
    match key:
        case 'relu':
            return nn.ReLU()
        case 'gelu':
            return nn.GELU()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'tanh':
            return nn.Tanh()
        case 'mish':
            return nn.Mish()
        case 'none':
            return nn.Identity()
        case _:
            raise NotImplementedError(f'Unsupported activation: {key}')

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG2 = np.log(2)

def squashed_gauss_rsample(mu, log_std):
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)
    pi_distribution = Normal(mu, std)
    action = pi_distribution.rsample()
    logprob = pi_distribution.log_prob(action).sum(dim=-1)
    # Refers to OpenAI's spiningup implementation https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    logprob -= (2*(LOG2 - action - F.softplus(-2 * action))).sum(dim=-1)
    return torch.tanh(action), logprob

def create_mlp(sizes, activation, output_activation=None, layer_norm=False):
    not_final = output_activation is None
    if output_activation is None: output_activation = activation
    layers = []
    for i in range(1, len(sizes)):
        act = activation if i < len(sizes) - 1 else output_activation
        if layer_norm and (not_final or i != len(sizes) - 1):
            layers += [nn.Linear(sizes[i-1], sizes[i]), nn.LayerNorm(sizes[i]), create_activation(act)]
        else:
            layers += [nn.Linear(sizes[i-1], sizes[i]), create_activation(act)]
    return nn.Sequential(*layers)

def create_unihid_mlp(in_size, out_size, hidden_size, hidden_layers, activation, output_activation=None, layer_norm=False):
    sizes = [in_size, *[hidden_size] * hidden_layers, out_size]
    return create_mlp(sizes, activation, output_activation, layer_norm)

def create_group_mlp(sizes, activation, output_activation=None, layer_norm=False, groups=2):
    not_final = output_activation is None
    if output_activation is None: output_activation = activation
    layers = []
    for i in range(1, len(sizes)):
        act = activation if i < len(sizes) - 1 else output_activation
        layers += [nn.Conv1d(groups*sizes[i-1], groups*sizes[i], 1, groups=groups), create_activation(act)]
        if layer_norm and (not_final or i != len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: np.ndarray
    alphas: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    log_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray
    posterior_variance: np.ndarray
    posterior_log_variance_clipped: np.ndarray
    posterior_mean_coef1: np.ndarray
    posterior_mean_coef2: np.ndarray

    @staticmethod
    def from_beta(betas: np.ndarray):
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            betas, alphas, alphas_cumprod, alphas_cumprod_prev,
            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
            posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas

    @staticmethod
    def vp_from_T(timesteps: int):
        betas = BetaScheduleCoefficients.vp_beta_schedule(timesteps)
        return BetaScheduleCoefficients.from_beta(betas)

# @dataclass(frozen=True)
# class BetaScheduleCoefficients:
#     betas: torch.Tensor
#     alphas: torch.Tensor
#     alphas_cumprod: torch.Tensor
#     alphas_cumprod_prev: torch.Tensor
#     sqrt_alphas_cumprod: torch.Tensor
#     sqrt_one_minus_alphas_cumprod: torch.Tensor
#     log_one_minus_alphas_cumprod: torch.Tensor
#     sqrt_recip_alphas_cumprod: torch.Tensor
#     sqrt_recipm1_alphas_cumprod: torch.Tensor
#     posterior_variance: torch.Tensor
#     posterior_log_variance_clipped: torch.Tensor
#     posterior_mean_coef1: torch.Tensor
#     posterior_mean_coef2: torch.Tensor
#
#     @staticmethod
#     def from_beta(betas: torch.Tensor):
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, dim=0)
#         alphas_cumprod_prev = torch.cat([torch.ones(1, device=betas.device), alphas_cumprod[:-1]])
#
#         sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#         sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
#         log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
#         sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
#         sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
#
#         posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#         posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
#         posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
#         posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
#
#         return BetaScheduleCoefficients(
#             betas, alphas, alphas_cumprod, alphas_cumprod_prev,
#             sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
#             sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
#             posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
#         )
#
#     @staticmethod
#     def vp_beta_schedule(timesteps: int, device="cpu"):
#         t = torch.arange(1, timesteps + 1, device=device)
#         T = timesteps
#         b_max = 10.
#         b_min = 0.1
#         alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
#         betas = 1 - alpha
#         return betas
#
#     @staticmethod
#     def from_T(timesteps: int, device='cpu'):
#         betas = BetaScheduleCoefficients.vp_beta_schedule(timesteps, device)
#         return BetaScheduleCoefficients.from_beta(betas)

