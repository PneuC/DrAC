import glob
import os
import numpy as np
import pandas as pd
from myutils.filesys import gp, load_yaml, save_csv
from analysis.test import *


def eval_robustness(folder, device='cuda:0'):
    from envs.mgmaze.point_maze import register_point_maze
    register_point_maze()
    os.makedirs(gp(folder, 'robustness'), exist_ok=True)

    path = gp(folder, 'robustness', 'removing_robustness.csv')
    if not os.path.exists(path):
        config = load_yaml(gp(folder, 'config.yaml'))
        num_goals = 8 if config['maze_map'] == 'hard' else 4
        cols = ['SR-mean', 'SR-median', 'SR-std', 'SR-min', 'SR-max']
        data = []
        for n in range(1, num_goals):
            sr = test_removal_robustness(folder, num_removal=n, device=device)
            data.append([np.mean(sr), np.median(sr), np.std(sr), min(sr), max(sr)])
        df = pd.DataFrame(data, columns=cols, index=list(range(1, num_goals)))
        df.to_csv(path)

    path = gp(folder, 'robustness', 'obstacle_robustness.csv')
    if not os.path.exists(path):
        cols = ['reward', 'reachable_modes', 'multi_goal_score']
        data = []
        num_obstacles = list(range(1, 11))
        for n in num_obstacles:
            results = test_obstacle_robustness(folder, n, device=device)
            data.append(
                [np.mean(results['reward']), np.mean(results['reachable_modes']), np.mean(results['multi_goal_score'])]
            )
        df = pd.DataFrame(data, columns=cols, index=num_obstacles)
        df.to_csv(path)
    pass

def eval_robustness_throughout_learning(folder, device='cuda:0'):
    from envs.mgmaze.point_maze import register_point_maze
    register_point_maze()
    os.makedirs(gp(folder, 'robustness'), exist_ok=True)

    path = gp(folder, 'robustness', 'by_step_removing.csv')
    if not os.path.exists(path):
        config = load_yaml(gp(folder, 'config.yaml'))
        num_goals = 8 if config['maze_map'] == 'hard' else 4
        cols = ['step', 'SR-mean', 'SR-median', 'SR-std', 'SR-min', 'SR-max']
        data = []
        ckpts = [int(os.path.split(ckpt)[1][:-3]) for ckpt in glob.glob(gp(folder, 'checkpoints', '*.pt'))]
        for t in sorted(ckpts):
            sr = test_removal_robustness(folder, num_removal=num_goals // 2, ckpt=t, device=device)
            data.append([t, np.mean(sr), np.median(sr), np.std(sr), min(sr), max(sr)])
        save_csv(path, cols, data)

    path = gp(folder, 'robustness', 'by_step_obstacle.csv')
    if not os.path.exists(path):
        cols = ['step', 'reward', 'reachable_modes', 'multi_goal_score']
        data = []
        ckpts = [int(os.path.split(ckpt)[1][:-3]) for ckpt in glob.glob(gp(folder, 'checkpoints', '*.pt'))]
        for t in sorted(ckpts):
            results = test_eval_mode_robustness(folder, t, device=device)
            data.append(
                [t, np.mean(results['reward']), np.mean(results['reachable_modes']), np.mean(results['multi_goal_score'])]
            )
        save_csv(path, cols, data)


if __name__ == '__main__':
    eval_robustness_throughout_learning('data/PointMaze/Simple/DrAC-test')
