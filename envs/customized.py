import numpy as np
import gymnasium as gym
import itertools as itt
from matplotlib.patches import Rectangle


class MultiGoal(gym.Env):
    distance = 10.0
    reward_scale = 1.0
    tolerance = 1.0

    def __init__(self, sparse=False):
        super().__init__()
        d = self.distance
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-2*d, high=2*d, shape=(2,))
        self.goals = np.array([[d, 0], [-d, 0], [0, d], [0, -d]])
        self.pos = np.zeros([2], dtype=np.float32)
        self.rng = np.random.default_rng()
        self.sparse = sparse
        self.num_goals = 4
        self.d_before = d

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.zeros([2])
        self.d_before = self.distance
        return self.get_obs(), {'pos': self.pos.copy(), 'distance': self.distance, 'target': 0}

    def step(self, action):
        reached = self.detect(action)
        self.move(action)
        d_after = self.metric(self.pos, self.goals[0])
        for goal in self.goals[1:]:
            d_after = min(d_after, self.metric(self.pos, goal))
        reward = 0 if self.sparse else self.d_before - d_after
        if reached:
            reward += 10 
        terminated, truncated = reached > 0, np.abs(self.pos).max() > 2 * self.distance
        self.d_before = d_after
        info = {'pos': self.pos.copy(), 'distance': d_after, 'target': reached}
        return self.get_obs(), reward * self.reward_scale, terminated, truncated, info

    def move(self, action):
        self.pos += action

    def detect(self, action):
        reached = 0
        for i, goal in enumerate(self.goals):
            checkpoints = np.linspace(self.pos+action, self.pos, 10, False)
            distances = self.metric(checkpoints, goal)
            if distances.min() <= self.tolerance:
                reached = i + 1
                break
        return reached

    def get_obs(self):
        return self.pos.copy().astype(np.float32)

    @staticmethod
    def metric(a, b):
        return np.abs(a - b).max(axis=-1)
        # return np.linalg.norm(a - b, axis=-1)

    def plot(self, ax, placeholder=None):
        l = 2 * self.tolerance
        for x, y in itt.chain(self.goals):
            ax.add_patch(Rectangle((x-self.tolerance, y-self.tolerance), l, l, fill=False))


class MultiGoalObstacle(MultiGoal):
    obst_size = 2.0

    def move(self, action):
        a, b = 0.5 * self.distance, self.obst_size
        x, y = self.pos
        xp, yp = self.pos + action
        if abs(x) <= a and abs(xp) >= a:
            rho = (a - abs(x)) / (abs(xp) - abs(x))
            y_cross = (abs(yp) - abs(y)) * rho + abs(y)
            if y_cross < b:
                self.pos[0] = x + (xp - x) * rho
                self.pos[1] = y + (yp - y) * rho
                return
        self.pos += action

    def plot(self, ax):
        super().plot(ax)
        a, b = 0.5 * self.distance, self.obst_size
        ax.plot([-a, -a], [-b, b], c='black')
        ax.plot([a, a], [-b, b], c='black')


class MultiGoalEuclidean(MultiGoal):
    @staticmethod
    def metric(a, b):
        return np.linalg.norm(a - b, axis=-1)


class MultiGoalObstacleEuclidean(MultiGoalObstacle):
    @staticmethod
    def metric(a, b):
        return np.linalg.norm(a - b, axis=-1)


def register_customized():
    gym.register('MultiGoal-v0', entry_point=MultiGoal, max_episode_steps=50)
    gym.register('MultiGoalObstacle-v0', entry_point=MultiGoalObstacle, max_episode_steps=50)
    gym.register('MultiGoal-Euclidean-v0', entry_point=MultiGoalEuclidean, max_episode_steps=50)
    gym.register('MultiGoalObstacle-Euclidean-v0', entry_point=MultiGoalObstacleEuclidean, max_episode_steps=50)

