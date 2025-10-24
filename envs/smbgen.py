import math
import random
import gymnasium as gym
import numpy as np
import torch
from src.gan.gans import get_decoder, nz, process_onehot
from myutils.datastruct import RingQueue
from myutils.mathtools import a_clip
from smb.level import *


W = MarioLevel.seg_width


class MultiFacet:
    g = 0.14

    def __init__(self, n=5):
        self.n = n

    def eval(self, seg, history):
        rew = 0.0
        r_sum = 0
        for i in range(self.n):
            s, e = history.w - (i+1) * W, history.w - i * W
            div = tile_pattern_js_div(seg, history[:, s:e])
            r = 1 - i / (self.n + 1)
            r_sum += r
            rew += a_clip(div, self.g, r)
        return rew / r_sum


class MarioPuzzle:
    lb, ub = 0.26, 0.94
    stride = 8
    constraint_weight = 10

    def __init__(self, m=3, n=5):
        self.m = m
        self.n = n

    def eval(self, seg, history):
        divs = []
        for shift in range(0, (self.n-1)*W, self.stride):
            s, e = history.w - shift - W, history.w - shift
            divs.append(tile_pattern_kl_div(seg, history[:, s:e]))
        divs.sort()
        rew = np.mean(divs[:self.m])
        mean_div = np.mean(divs)
        # Reshape reward
        penalty = 0
        if mean_div < self.lb:
            penalty = (mean_div - self.lb) ** 2
        elif mean_div > self.ub:
            penalty = (mean_div - self.ub) ** 2
        # Reshape reward (because KL is exponential scale)
        if penalty > 1:
            penalty = math.log(penalty) + 1
        rew -= self.constraint_weight * penalty
        return rew


class MarioGenerationEnv(gym.Env):
    def __init__(self, style='MultiFacet', eplen: int=50, device='cuda:0'):
        match style:
            case 'MultiFacet':
                self.rfunc = MultiFacet()
            case 'MarioPuzzle':
                self.rfunc = MarioPuzzle()
            case _:
                raise NotImplementedError(f'Unknow Mario level generation style: {style}')
        self.init_vecs = np.load(gp('smb/init_latvecs.npy'))
        self.decoder = get_decoder(device=device)
        self.decoder.to(device)
        self.wd = self.rfunc.n
        self.eplen = eplen
        self.device = device
        self.action_space = gym.spaces.Box(-1, 1, (nz,))
        self.observation_space = gym.spaces.Box(-1, 1, (self.wd * nz,))
        self.lat_vecs = RingQueue(self.wd)
        self.history = None
        self.latest = None
        self.rng = random.Random()
        self.l = 0

    def step(self, action):
        self.lat_vecs.push(action)
        seg = self.get_seg(action)
        rew = self.rfunc.eval(seg, self.history)

        playable = check_playable(self.latest + seg)
        rew -= int(not playable)
        self.l += 1
        self.history = self.history + seg
        self.latest = seg
        trunct = self.l >= self.eplen
        lvl = str(self.history[:, (self.wd-1)*W:]) if trunct else None
        return self.get_ob(), rew, False, trunct, {'playable': playable, 'level': lvl}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        init_segs = []
        for _ in range(self.wd):
            i = self.rng.randrange(0, len(self.init_vecs))
            z = self.init_vecs[i]
            seg = self.get_seg(z)
            self.lat_vecs.push(z)
            init_segs.append(seg)

        self.latest = init_segs[-1]
        self.history = lvl_sum(init_segs)
        self.l = 0
        return self.get_ob(), {'playable': True, 'level': None}

    def get_ob(self):
        return np.concat(self.lat_vecs.to_list())

    def get_seg(self, action):
        z = torch.tensor(action, device=self.device, dtype=torch.float).view(1, nz, 1, 1)
        return process_onehot(self.decoder(z))


def register_smbgen():
    gym.register('MarioLevelGen', MarioGenerationEnv, max_episode_steps=50)



class VecMarioGenerationEnv(gym.Env):
    def __init__(self, style='MultiFacet', num_envs=16, eplen: int=50, device='cuda:0'):
        match style:
            case 'MultiFacet':
                self.rfunc = MultiFacet()
            case 'MarioPuzzle':
                self.rfunc = MarioPuzzle()
            case _:
                raise NotImplementedError(f'Unknow Mario level generation style: {style}')
        self.init_vecs = np.load(gp('smb/init_latvecs.npy'))
        self.decoder = get_decoder(device=device)
        self.decoder.to(device)
        self.wd = self.rfunc.n
        self.eplen = eplen
        self.device = device
        self.action_space = gym.spaces.Box(-1, 1, (num_envs, nz,))
        self.observation_space = gym.spaces.Box(-1, 1, (num_envs, nz,))
        self.lat_vecs = [RingQueue(self.wd) for _ in range(num_envs)]
        self.history = [None] * num_envs
        self.latest = [None] * num_envs
        self.rng = random.Random()
        self.l = 0
        self.num_envs = num_envs
        self.need_reset = True
        self.stop_reward = False

    def step(self, action):
        if self.need_reset:
            o, info = self.reset()
            rewards = [0] * self.num_envs
            done = [False] * self.num_envs
            trunct = [False] * self.num_envs
            return o, rewards, done, trunct, info
        rewards = []
        segs = self.get_seg(action)
        for i in range(self.num_envs):
            self.lat_vecs[i].push(action[i])
            if self.stop_reward:
                rew = 0
            else:
                rew = self.rfunc.eval(segs[i], self.history[i])
                playable = check_playable(self.latest[i] + segs[i])
                rew -= int(not playable)
            rewards.append(rew)
            self.history[i] = self.history[i] + segs[i]
            self.latest[i] = segs[i]
        self.l += 1
        trunct = [self.l >= self.eplen]  * self.num_envs
        lvls = [str(item[:, (self.wd-1)*W:]) for item in self.history] if trunct else [None] * self.num_envs
        if self.l >= self.eplen:
            self.need_reset = True
        return self.get_ob(), rewards, [False] * self.num_envs, trunct, {'level': lvls}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        for i in range(self.num_envs):
            # init_segs = []
            for _ in range(self.wd):
                j = self.rng.randrange(0, len(self.init_vecs))
                z = self.init_vecs[j]
                self.lat_vecs[i].push(z)
                # init_segs.append(seg)
            zs = np.stack(self.lat_vecs[i].to_list(), axis=0)
            init_segs =  self.get_seg(zs)
            self.latest[i] = init_segs[-1]
            self.history[i] = lvl_sum(init_segs)
        self.need_reset = False
        self.l = 0
        return self.get_ob(), {'level': [None] * self.num_envs}

    def get_ob(self):
        states = [np.concat(queue.to_list()) for queue in self.lat_vecs]
        return np.stack(states, axis=0)

    def get_seg(self, action):
        n = len(action)
        z = torch.tensor(action, device=self.device, dtype=torch.float).view(n, nz, 1, 1)
        return process_onehot(self.decoder(z))


if __name__ == '__main__':
    register_smbgen()
    env = gym.make('MarioLevelGen')
    env.reset()

    for _ in range(100):
        o, r, d, trunc, info = env.step(env.action_space.sample())
        print(o.shape, r, 'playable:', info['playable'])
