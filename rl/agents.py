import torch
from abc import abstractmethod
from myutils.filesys import gp
from rl.strategies import PeriodicTrigger


class ActCrtAgent:
    log_keys = ('policy_loss', 'critic_loss')

    def __init__(self, actor, critic, device='cpu'):
        self.actor = actor
        self.critic = critic
        self.device = device
        self.to(device)
        print("Create a %s agent" % self.__class__.__name__)
        print("Actor NN Architecture:\n", str(self.actor.net))
        print("Critic NN Architecture:\n", str(self.critic.net))

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.device = device

    def make_decision(self, obs):
        a = self.actor.forward(
            torch.tensor(obs, dtype=torch.float, device=self.device),
            grad=False
        )
        return a.cpu().numpy()

    def step_callback(self, *args, **kwargs):
        return

    def ep_callback(self, *args, **kwargs):
        return

    @abstractmethod
    def update(self, batch) -> dict:
        pass

    def save(self, path):
        ckpt = {
            'actor': self.actor.net.state_dict(), 
            'critic': self.critic.net.state_dict(), 
        }
        torch.save(ckpt, gp(path))

    def load(self, path, device='cpu'):
        ckpt = torch.load(gp(path), device, weights_only=True)
        self.actor.net.load_state_dict(ckpt['actor'])
        self.critic.net.load_state_dict(ckpt['critic'])
        self.to(device)


class SAC(ActCrtAgent):
    log_keys = ('policy_loss', 'critic_loss', 'temperature', 'entropy')

    def make_decision(self, obs, deterministic=False):
        a, _ = self.actor.forward(
            torch.tensor(obs, dtype=torch.float, device=self.device),
            grad=False, deterministic=deterministic
        )
        return a.cpu().numpy()

    def update(self, batch, importance=None):
        obs, acts, rews, ops, dones = batch
        if importance is None:
            importance = torch.ones_like(rews)
        critic_loss = self.critic.update_vf(self.actor, obs, acts, rews, ops, dones, importance)
        self.critic.update_tar_net()
        policy_loss, logps = self.actor.update_policy(self.critic, obs, importance)
        temperature, entropy = self.actor.update_alpha(logps, importance)
        info = {
            'policy_loss': policy_loss, 'critic_loss': critic_loss, 
            'temperature': temperature, 'entropy': entropy
        }
        return info

    def compute_error(self, batch):
        obs, acts, rews, ops, dones = batch
        return self.critic.compute_error(self.actor, obs, acts, rews, ops, dones)


class DrACV0(ActCrtAgent):
    log_keys = ('policy_qmean', 'policy_reg', 'critic_loss')

    def update(self, batch, importance=None):
        obs, acts, rews, ops, dones = batch
        if importance is None:
            importance = torch.ones_like(rews)
        critic_loss = self.critic.update_vf(self.actor, obs, acts, rews, ops, dones, importance)
        self.critic.update_tar_net()
        qmean, reg = self.actor.update_policy(self.critic, obs, importance)
        info = {
            'policy_qmean': qmean, 'policy_reg': reg, 'critic_loss': critic_loss
        }
        return info

    def compute_error(self, batch):
        obs, acts, rews, ops, dones = batch
        return self.critic.compute_error(self.actor, obs, acts, rews, ops, dones)

    def step_callback(self, *args, **kwargs):
        if self.actor.alpha.__class__.__name__ == 'LinearScheduledCoefficient':
            self.actor.alpha.update(kwargs['completion_rate'])


class DrAC(DrACV0):
    log_keys = ('policy_qmean', 'policy_reg', 'critic_loss', 'alpha')

    def __init__(self, actor, critic, alpha_update_repeat=1, device='cpu'):
        super().__init__(actor, critic, device)
        self.alpha_update_repeat = alpha_update_repeat

    def update(self, batch, importance=None):
        obs, acts, rews, ops, dones = batch
        if importance is None:
            importance = torch.ones_like(rews)
        critic_loss = self.critic.update_vf(self.actor, obs, acts, rews, ops, dones, importance)
        self.critic.update_tar_net()
        qmean, regs = self.actor.update_policy(self.critic, obs, importance)
        alpha = 0
        for _ in range(self.alpha_update_repeat):
            alpha = self.actor.update_alpha(regs, importance)
        info = {
            'policy_qmean': qmean, 'policy_reg': regs.mean().item(), 'critic_loss': critic_loss, 'alpha': alpha
        }
        return info

    def step_callback(self, *args, **kwargs):
        if 'Scheduled' in self.actor.beta.__class__.__name__:
            self.actor.beta.update(kwargs['completion_rate'])


class DACER(ActCrtAgent):
    log_keys = ('policy_loss', 'critic_loss', 'alpha', 'entropy')
    def __init__(self, actor, critic, policy_delay=2, alpha_update_itv=10000, device='cpu'):
        super().__init__(actor, critic, device)
        self.policy_update_trigger = PeriodicTrigger(policy_delay)
        self.alpha_update_trigger = PeriodicTrigger(alpha_update_itv)

    def make_decision(self, obs, deterministic=False):
        a = self.actor.forward(
            torch.tensor(obs, dtype=torch.float, device=self.device),
            grad=False, deterministic=deterministic
        )
        return a.cpu().numpy()

    def update(self, batch, importance=None):
        obs, acts, rews, ops, dones = batch
        if importance is None:
            importance = torch.ones_like(rews)
        critic_loss = self.critic.update_vf(self.actor, obs, acts, rews, ops, dones, importance)
        self.critic.update_tar_net()

        if self.policy_update_trigger.step():
            policy_loss = self.actor.update_policy(self.critic, obs, importance)
            policy_loss *= self.policy_update_trigger.itv
        else:
            policy_loss = 0

        if self.alpha_update_trigger.step():
            entropy = self.actor.estimate_entropy()
            self.actor.update_alpha(entropy, importance)

        return {'policy_loss': policy_loss, 'critic_loss': critic_loss, 'entropy': self.actor.entropy, 'alpha': self.actor.alpha.value()}
