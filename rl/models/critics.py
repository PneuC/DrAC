import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from rl.strategies import polyak_update


class DualClipCritic:
    def __init__(self, net, gamma=0.99, tau=0.005, lr=3e-4, device='cpu'):
        self.net = net
        self.tar_net = deepcopy(self.net)
        self.tar_net.requires_grad_(False)
        self.opt = Adam(self.net.parameters(), lr)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.to(device)

    def to(self, device):
        self.net.to(device)
        self.tar_net.to(device)
        self.device = device

    def forward(self, obs, acts, grad=True, tar=False):
        net = self.tar_net if tar else self.net
        if grad:
            return net.predict(obs, acts)
        with torch.no_grad():
            return net.predict(obs, acts)

    def compute_target(self, actor, rews, ops, dones):
        aps = actor.forward(ops, grad=False)
        vps = self.forward(ops, aps, tar=True, grad=False)
        y = rews + self.gamma * (1 - dones) * vps
        return y

    def compute_error(self, actor, obs, acts, rews, ops, dones):
        y = self.compute_target(actor, rews, ops, dones)
        q = self.forward(obs, acts, grad=False)
        return (y - q).abs().cpu().numpy()

    def update_vf(self, actor, obs, acts, rews, ops, dones, importance):
        self.opt.zero_grad()
        y = self.compute_target(actor, rews, ops, dones)
        q1, q2 = self.net(obs, acts)
        loss = F.mse_loss(q1, y, reduction='none') + F.mse_loss(q2, y, reduction='none')
        loss = (loss * importance).mean()
        loss.backward()
        self.opt.step()
        return loss.item()

    def update_tar_net(self):
        polyak_update(self.net.parameters(), self.tar_net.parameters(), self.tau)


class SoftDualClipCritic(DualClipCritic):
    def compute_target(self, actor, rews, ops, dones):
        aps, ent_est = actor.forward(ops, grad=False)
        vps = self.forward(ops, aps, tar=True, grad=False)
        y = rews + self.gamma * (1 - dones) * (vps - actor.alpha * ent_est)
        return y


class LatentDualClipCritic(DualClipCritic):
    def __init__(self, net, qt_samples=1, gamma=0.99, tau=0.005, lr=3e-4, device='cpu'):
        super().__init__(net, gamma, tau, lr, device)
        self.qt_samples = qt_samples

    def compute_target(self, actor, rews, ops, dones):
        with torch.no_grad():
            aps, xps, yps = actor.sample_actions(ops)
            vps = self.forward(ops, aps, tar=True)
            regs = actor.calc_reg_from_samples(xps, yps)
        return rews + self.gamma * (1 - dones) * (vps + actor.alpha * regs)


class TriRefinedDistributionalCritic(DualClipCritic):
    xi = 3
    eps = 0.1    # DACER repo use shared eps at 0.1

    def __init__(self, net, gamma=0.99, tau=0.005, lr=3e-4, device='cpu'):
        super().__init__(net, gamma, tau, lr, device)
        self.avg_std1, self.avg_std2 = None, None

    def forward(self, obs, acts, grad=True, tar=False):
        if grad:
            return self.net.predict(obs, acts, stoc=False)
        with torch.no_grad():
            return self.net.predict(obs, acts, stoc=False)

    def compute_target(self, actor, rews, ops, dones):
        aps = actor.forward(ops, grad=False)
        with torch.no_grad():
            (mu1, mu2), (std1, std2) = self.tar_net.forward(ops, aps)
            vp_deterministic = torch.minimum(mu1, mu2)
            z1 = mu1 + torch.clip(torch.randn_like(mu1), -3.0, 3.0) * std1
            z2 = mu2 + torch.clip(torch.randn_like(mu2), -3.0, 3.0) * std2
            diff = mu1 - mu2
            vp_stochastic = torch.where(diff < 0, z1, z2)
            y_deterministic = rews + self.gamma * (1 - dones) * vp_deterministic
            y_stochastic = rews + self.gamma * (1 - dones) * vp_stochastic
        return y_deterministic, y_stochastic

    def update_vf(self, actor, obs, acts, rews, ops, dones, importance):
        self.opt.zero_grad()
        (mu1, mu2), (std1, std2) = self.net(obs, acts)
        # Update the running estimation of std
        avg_std1 = std1.detach().mean().item()
        avg_std2 = std2.detach().mean().item()
        if self.avg_std1 is None:
            self.avg_std1, self.avg_std2 = avg_std1, avg_std2
        else:
            self.avg_std1 = self.tau * avg_std1 + (1 - self.tau) * self.avg_std1
            self.avg_std2 = self.tau * avg_std2 + (1 - self.tau) * self.avg_std2
        # Calculate loss
        yd, ys = self.compute_target(actor, rews, ops, dones)
        mut_mu, mut_std = self.__calc_loss_multipliers(mu1, std1, yd, ys, self.avg_std1)
        loss1 = -(mut_mu * mu1 + mut_std * std1) * importance
        mut_mu, mut_std = self.__calc_loss_multipliers(mu2, std2, yd, ys, self.avg_std2)
        loss2 = -(mut_mu * mu2 + mut_std * std2) * importance
        # Back-propagate & descend
        w1, w2 = self.avg_std1 ** 2 + self.eps, self.avg_std2 ** 2 + self.eps
        loss = w1 * loss1.mean() + w2 * loss2.mean()
        loss.backward()
        self.opt.step()
        return loss.item()

    def __calc_loss_multipliers(self, mu, std, yd, ys, avg_std):
        sg_mu, sg_std = mu.detach(), std.detach()
        mut_mu = (yd - sg_mu) / (sg_std ** 2 + self.eps)
        ys_clip = sg_mu + torch.clamp(ys - sg_mu, -self.xi * avg_std, self.xi * avg_std)
        mut_std = ((ys_clip - sg_mu) ** 2 - sg_std ** 2) / (sg_std ** 3 + self.eps)
        return mut_mu, mut_std

    def update_tar_net(self):
        polyak_update(self.net.parameters(), self.tar_net.parameters(), self.tau)
