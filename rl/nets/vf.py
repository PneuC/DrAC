import torch
from torch import nn
from rl.nets.base import *


class QValueMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, hidden_layers=2, activation='relu', layer_norm=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # self.main = create_mlp([obs_dim + act_dim, *hidden_sizes, 1], activation, 'none', layer_norm)
        self.main = create_unihid_mlp(obs_dim + act_dim, 1, hidden_size, hidden_layers, activation, 'none', layer_norm)

    def forward(self, obs, act):
        return self.main(torch.cat([obs, act], dim=-1)).squeeze()


class DoubleQMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, hidden_layers=2, activation='relu', layer_norm=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # self.qf1 = create_mlp([obs_dim + act_dim, *hidden_sizes, 1], activation, 'none', layer_norm)
        # self.qf2 = create_mlp([obs_dim + act_dim, *hidden_sizes, 1], activation, 'none', layer_norm)
        self.qf1 = create_unihid_mlp(obs_dim + act_dim, 1, hidden_size, hidden_layers, activation, 'none', layer_norm)
        self.qf2 = create_unihid_mlp(obs_dim + act_dim, 1, hidden_size, hidden_layers, activation, 'none', layer_norm)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.qf1(x).squeeze(), self.qf2(x).squeeze()

    def predict(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.minimum(q1, q2)


class DoubleGaussianMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, hidden_layers=2, activation='relu', layer_norm=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.qf1 = create_unihid_mlp(obs_dim + act_dim, 2, hidden_size, hidden_layers, activation, 'none', layer_norm)
        self.qf2 = create_unihid_mlp(obs_dim + act_dim, 2, hidden_size, hidden_layers, activation, 'none', layer_norm)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        o1, o2 = self.qf1(x), self.qf2(x)
        mu = (o1[..., 0], o2[..., 0])
        std = (F.softplus(o1[..., 1]), F.softplus(o2[..., 1]))
        return mu, std

    def predict(self, obs, act, stoc=False):
        mu, std = self.forward(obs, act)
        if stoc:
            q_stoc1 = Normal(mu[0], std[0]).sample()
            q_stoc2 = Normal(mu[1], std[1]).sample()
            return torch.minimum(q_stoc1, q_stoc2)
        else:
            return torch.minimum(*mu)
