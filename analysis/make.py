import torch
import random
import numpy as np
from analysis.viz import *
from analysis.test import rollout_pos_trajs
from rl.rlfactory import load_agent


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def make_trajs_fig(agent, env, fname='eval_trajs.png', title='Evaluation Trajectories', parallel=10, eps=50):
    trajs = rollout_pos_trajs(env, agent, eps, 0)
    viz_trajs(env.envs[0], trajs, f'{folder}/{fname}', title='Evaluation')

def make_final_trajs_fig(folder, fname='eval_trajs.png', title='Evaluation Trajectories', env=None, parallel=10, eps=50):
    agent, args = load_agent(folder)
    if env is None:
        import gymnasium as gym
        from envs.customized import register_customized
        register_customized()
        env = gym.make_vec(args.task, parallel, 'sync')
    trajs = rollout_pos_trajs(env, agent, eps, 0)
    viz_trajs(env.envs[0], trajs, f'{folder}/{fname}', title='Evaluation')



