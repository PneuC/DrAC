import math
import numpy as np
import gymnasium as gym
import itertools as itt
from rl.rlfactory import load_agent
from envs.mgmaze.maps import get_map
from envs.mgmaze.maze import get_removed_maps, add_random_obstacles
from smb.level import MarioLevel, lvl_hamming_dis


def std_rl_test(vec_env, agent, eps, seed=None, take_mean=True):
    buffer = [[] for _ in range(vec_env.num_envs)]
    rewards, eplens = [], []
    o, info = vec_env.reset(seed=seed)
    if 'target' in info.keys():
        modes = [0] * vec_env.envs[0].unwrapped.num_goals
    else:
        modes = None
    levels = [] if 'level' in info.keys() else None

    while len(rewards) < eps:
        a = agent.make_decision(o)
        o, r, terminated, truncated, info = vec_env.step(a)
        for traj, r in zip(buffer, r):
            traj.append(r)
        for i, (t1, t2) in enumerate(zip(terminated, truncated)):
            if t1 or t2:
                rewards.append(sum(buffer[i]))
                eplens.append(len(buffer[i]))
                buffer[i].clear()
                if modes is not None:
                    j = int(info['target'][i])
                    if j != 0:
                        modes[j-1] = modes[j-1] + 1
                if levels is not None:
                    levels.append(MarioLevel(info['level'][i]))
                    pass
                if len(rewards) >= eps:
                    break
    if take_mean:
        results = {
            'reward': np.mean(rewards), 'reward-std': np.std(rewards),
            'eplen': np.mean(eplens), 'eplen-std': np.std(eplens)
        }
    else:
        results = {'reward': np.array(rewards), 'eplen': np.array(eplens)}
    if modes is not None:
        multi_goal_score = 0
        reachable_modes = 0
        for i in range(len(modes)):
            if modes[i] > 0:
                reachable_modes += 1
                multi_goal_score += min(modes[i] / eps, 1 / len(modes))
        results.update(reachable_modes=reachable_modes, multi_goal_score=multi_goal_score)
    if levels is not None:
        results['levels'] = levels
    return results

def smbgen_diversity_test(vec_env, agent, pairs=150, seed=None):
    levels = []
    o, info = vec_env.reset(seed=seed)
    vec_env.stop_reward = True
    vec_env.eplen = 20
    while len(levels) < 2 * pairs:
        a = agent.make_decision(o)
        o, r, terminated, truncated, info = vec_env.step(a)
        for i, (t1, t2) in enumerate(zip(terminated, truncated)):
            if t1 or t2:
                levels.append(MarioLevel(info['level'][i]))
                if len(levels) == 2 * pairs:
                    break
    distances = np.array([lvl_hamming_dis(a, b) for a, b in itt.batched(levels, 2)])
    gmean = math.exp(np.log(distances + 1e-3).mean())
    vec_env.stop_reward = False
    vec_env.eplen = 50
    return {'avg-distance': np.mean(distances), 'gmean-distance': gmean}

def rollout_pos_trajs(envs, agent, eps, seed=None):
    o, info = envs.reset(seed=seed)
    buffer = [[pos] for pos in info['pos']]
    trajs = []
    while len(trajs) < eps:
        a = agent.make_decision(o)
        o, _, terminated, truncated, info = envs.step(a)
        for traj, pos in zip(buffer, info['pos']):
            traj.append(pos)
        for i, (t1, t2) in enumerate(zip(terminated, truncated)):
            if t1 or t2:
                trajs.append(np.stack(buffer[i]))
                buffer[i] = []
                if len(trajs) >= eps:
                    break
    return trajs

def test_n_trial_success_rate(env, agent, trials=5, repeats=100, num_removal=None):
    o, _ = env.reset(seed=0)
    reaches = []
    while len(reaches) < trials * repeats:
        a = agent.make_decision(o)
        o, _, terminated, truncated, info = env.step(a)
        for i, (t1, t2) in enumerate(zip(terminated, truncated)):
            if t1 or t2:
                reaches.append(int(info['target'][i]))
    success_count = 0
    for i in range(repeats):
        window = slice(i * trials, (i + 1) * trials)
        if num_removal is None:
            success_count += int(any(reached != 0 for reached in reaches[window]))
        else:
            R = len(set(reaches[window]) - {0})
            N = env.envs[0].unwrapped.num_goals
            if R >= N - num_removal:
                success_count += 1
            else:
                success_count += 1 - math.comb(N-R, num_removal) / math.comb(N, num_removal)
    success_rate = success_count / repeats
    return success_rate

def test_removal_robustness(maze_map, agent, num_removal, trials=5, repeats=100):
    success_rates = []
    env = gym.make_vec('MultiGoalPointMaze', num_envs=16, maze_map=get_map(maze_map))
    env.reset(seed=0)
    success_rate = test_n_trial_success_rate(env, agent, trials, repeats, num_removal)
    success_rates.append(success_rate)
    env.close()
    return np.mean(success_rates)

def test_obstacle_robustness(folder, num_obstacles, repeats_per_goal=20, num_maps=30, device='cuda:0'):
    agent, args = load_agent(folder, device=device)
    maze_map = get_map(args.maze_map)
    perturbed_maps = [add_random_obstacles(maze_map, num_obstacles) for _ in range(num_maps)]
    results = {'reward': [], 'reachable_modes': [], 'multi_goal_score': []}
    for mp in perturbed_maps:
        env = gym.make_vec(args.task, num_envs=16, maze_map=mp)
        n_goal = env.envs[0].unwrapped.num_goals
        info = std_rl_test(env, agent, repeats_per_goal * n_goal)
        results['reward'].append(info['reward'])
        results['reachable_modes'].append(info['reachable_modes'])
        results['multi_goal_score'].append(info['multi_goal_score'])
        print([f'{k}: {v}' for k, v in info.items()])
    return results

def test_eval_mode_robustness(maze_map, agent, trials=5, repeats=100):
    env = gym.make_vec('MultiGoalPointMaze', num_envs=16, maze_map=get_map(maze_map), maze_eval_mode=True)
    env.reset(seed=0)
    success_rate = test_n_trial_success_rate(env, agent, trials, repeats)
    env.close()
    return success_rate

