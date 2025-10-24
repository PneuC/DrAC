import math
from importlib.metadata import distributions

import torch
import numpy as np
from torch.optim import Adam
from rl.nets.auxiliary import LearnableCoeffient


class BaseNNActor:
    def __init__(self, net, lr=3e-4):
        self.net = net
        self.optimizer = Adam(self.net.parameters(), lr)
        self.device = 'cpu'

    def to(self, device):
        self.net.to(device)
        self.device = device

    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad_step(self):
        self.optimizer.step()

    def get_nn_arch_str(self):
        return str(self.net) + '\n'

    def forward(self, obs, grad=True, **kwargs):
        if grad:
            return self.net(obs, **kwargs)
        with torch.no_grad():
            return self.net(obs, **kwargs)


class SoftActor(BaseNNActor):
    def __init__(self, net, lr=3e-4, tar_ent=-8, alpha_lr=None):
        super(SoftActor, self).__init__(net, lr)
        self.alpha = LearnableCoeffient()
        self.alpha_optimizer = Adam(self.alpha.parameters(), lr if alpha_lr is None else alpha_lr)
        self.tar_ent = tar_ent

    def to(self, device):
        self.net.to(device)
        self.alpha.to(device)
        self.device = device

    def update_policy(self, critic, obs, importance):
        self.optimizer.zero_grad()
        critic.net.requires_grad_(False)
        acts, logps = self.forward(obs)
        qvalues = critic.forward(obs, acts)
        loss = (self.alpha(logps, grad=False) - qvalues)
        loss = (loss * importance).mean()
        loss.backward()
        self.optimizer.step()
        critic.net.requires_grad_(True)
        return loss.item(), logps.detach()

    def update_alpha(self, logps, importance):
        self.alpha_optimizer.zero_grad()
        # _, logps = self.forward(obs, grad=False)
        loss = -(self.alpha(logps + self.tar_ent) * importance).mean()
        loss.backward()
        self.alpha_optimizer.step()
        entropy = -logps.mean().item()
        return self.alpha.value(), entropy

    def forward(self, obs, grad=True, deterministic=False):
        if grad:
            return self.net(obs, deterministic)
        with torch.no_grad():
            return self.net(obs, deterministic)


class LatentActorV0(BaseNNActor):
    def __init__(self, net, lr=0.0003, alpha=0.15, pg_samples=8, reg_samples=16):
        # TODO: Multiple actions update
        super().__init__(net, lr)
        self.alpha = alpha
        self.pg_samples= pg_samples
        self.reg_samples = reg_samples

    def update_policy(self, critic, obs, importance):
        self.optimizer.zero_grad()
        critic.net.requires_grad_(False)
        if self.pg_samples == 1:
            acts = self.forward(obs)
            qvalues = critic.forward(obs, acts)
            regs = self.calc_reg(obs)
        else:
            obs_rept = obs.expand(self.pg_samples, *obs.shape)
            acts = self.forward(obs_rept)
            qvalues = critic.forward(obs_rept, acts).mean(dim=0)
            regs = self.calc_reg(obs)
        loss = -(qvalues + self.alpha * regs) * importance
        loss.mean().backward()
        self.optimizer.step()
        critic.net.requires_grad_(True)
        return qvalues.mean().item(), regs.detach()

    def calc_reg(self, obs):
        a = self.net.rept_sample(obs, self.reg_samples)
        b = self.net.rept_sample(obs, self.reg_samples)
        dists = torch.linalg.vector_norm(a - b, dim=-1)
        return torch.log(dists + 1e-5).mean(dim=0)


class LatentActor(LatentActorV0):
    def __init__(self, net, lr=0.0003, pg_samples=1, reg_samples=16, beta=0.5, init_alpha=1.0, alpha_lr=5e-3):
        super().__init__(net, lr, 0.0, pg_samples, reg_samples)
        self.reg_samples = reg_samples
        self.alpha = LearnableCoeffient(init_alpha)
        self.alpha_optimizer = Adam(self.alpha.parameters(), lr=alpha_lr)
        self.beta = beta
        self.scale = self.net.act_dim ** 0.5

    def to(self, device):
        self.net.to(device)
        self.alpha.to(device)
        self.device = device

    def update_policy(self, critic, obs, importance):
        self.optimizer.zero_grad()
        critic.net.requires_grad_(False)
        a, x, y = self.sample_actions(obs)
        qvalues = critic.forward(obs, a)
        regs = self.calc_reg_from_samples(x, y)
        loss = -(qvalues + self.alpha * regs) * importance
        loss.mean().backward()
        self.optimizer.step()
        critic.net.requires_grad_(True)
        return qvalues.mean().item(), regs.detach()
    #
    # def calc_reg(self, obs):
    #     a = self.net.rept_sample(obs, self.reg_samples)
    #     b = self.net.rept_sample(obs, self.reg_samples)
    #     dists = torch.linalg.vector_norm(a - b, dim=-1)
    #     return torch.log(dists + 1e-5).mean(dim=0)

    def calc_reg_from_samples(self, x, y):
        dists = torch.linalg.vector_norm(x - y, dim=-1)
        return torch.log(dists + 1e-5).mean(dim=0)

    def sample_actions(self, obs):
        samples = self.net.rept_sample(obs, 2 * self.reg_samples + 1)
        a = samples[0, ...]
        x = samples[1:self.reg_samples+1, ...]
        y = samples[self.reg_samples+1:, ...]
        return a, x, y

    def update_alpha(self, regs, importance):
        self.alpha_optimizer.zero_grad()
        loss = -(self.alpha(self.get_tar_divs() - regs) * importance).mean()
        loss.backward()
        self.alpha_optimizer.step()
        return self.alpha.value()

    def get_tar_divs(self):
        return np.log(float(self.beta) * self.scale)


class EfficientLatentActor(LatentActorV0):
    def __init__(self, net, lr=0.0003, pg_samples=1, reg_samples=16, beta=0.5, init_alpha=1.0):
        super().__init__(net, lr, 0.0, pg_samples, reg_samples)
        self.reg_samples = reg_samples
        self.alpha = LearnableCoeffient(init_alpha)
        self.alpha_optimizer = Adam(self.alpha.parameters(), 5e-3)
        self.beta = beta
        self.scale = self.net.act_dim ** 0.5

    def to(self, device):
        self.net.to(device)
        self.alpha.to(device)
        self.device = device

    def update_alpha(self, regs, importance):
        self.alpha_optimizer.zero_grad()
        loss = -(self.alpha(self.get_tar_divs() - regs) * importance).mean()
        loss.backward()
        self.alpha_optimizer.step()
        return self.alpha.value()

    def get_tar_divs(self):
        return np.log(float(self.beta) * self.scale)


class DACERActor(BaseNNActor):
    def __init__(self, net, lr=3e-4, tar_ent=-8, alpha_lr=3e-2, l=0.1, w0=3.0, ent_est_samples=200, ent_est_components=3, tanh_out=False):
        super().__init__(net, lr)
        self.cached_obs = None
        self.alpha = LearnableCoeffient(w0)
        self.alpha_optimizer = Adam(self.alpha.parameters(), lr if alpha_lr is None else alpha_lr)
        self.tar_ent = tar_ent
        self.l = l
        self.ent_est_samples = ent_est_samples
        self.ent_est_components = ent_est_components
        self.entropy = 0.0
        self.tanh_out = tanh_out

    def to(self, device):
        self.net.to(device)
        self.alpha.to(device)
        self.device = device

    def forward(self, obs, grad=True, **kwargs):
        if grad:
            action = self.net(obs)
        else:
            with torch.no_grad():
                action = self.net(obs)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            return torch.tanh(action) if self.tanh_out else torch.clip(action, -1, 1)
        else:
            if type(self.alpha) == float:
                a = action + self.alpha * self.l * torch.randn_like(action)
            else:
                a = action + self.alpha.value() * self.l * torch.randn_like(action)
            return torch.tanh(a) if self.tanh_out else torch.clip(a, -1, 1)

    def update_policy(self, critic, obs, importance):
        self.optimizer.zero_grad()
        acts = self.forward(obs)
        critic.net.requires_grad_(False)
        loss = -critic.forward(obs, acts)
        loss = (loss * importance).mean()
        loss.backward()
        self.optimizer.step()
        critic.net.requires_grad_(True)
        self.cached_obs = obs
        return loss.item()

    def estimate_entropy(self):  # (batch, sample, dim)
        from sklearn.mixture import GaussianMixture
        entropies = []
        actions = [self.forward(self.cached_obs) for _ in range(self.ent_est_samples)]
        sample_sets = torch.stack(actions, dim=1).detach().cpu().numpy()
        for sample_set in sample_sets:
            gmm = GaussianMixture(n_components=self.ent_est_components, covariance_type='full')
            gmm.fit(sample_set)
            weights = gmm.weights_
            dim_entropies = []
            for i in range(gmm.n_components):
                cov_matrix = gmm.covariances_[i]
                d = cov_matrix.shape[0]
                logp = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
                dim_entropies.append(logp)
            logp = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(dim_entropies))
            entropies.append(logp)
        # return torch.tensor(entropies, device=self.actions.device)
        return sum(entropies) / len(entropies)

    def update_alpha(self, entropy, importance):
        self.alpha_optimizer.zero_grad()
        # _, logps = self.forward(obs, grad=False)
        loss = -(self.alpha(self.tar_ent - entropy) * importance).mean()
        loss.backward()
        self.alpha_optimizer.step()
        self.entropy = entropy
        # print('Entropy:', entropy, 'alpha:', self.alpha.value())
