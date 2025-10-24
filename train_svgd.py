import gymnasium as gym
import argparse

from analysis.test import rollout_pos_trajs, test_removal_robustness, test_eval_mode_robustness
from analysis.viz import viz_trajs
from envs.factory import make_env
from myutils.filesys import gp, auto_dire, save_json, load_yaml, save_yaml
from myutils.fmt import now_datetime
from myutils.tools import fix_seed
from rl.loggers import CsvLogger
import torch
import os
# from torch.utils.tensorboard import SummaryWriter
# from stac.envs.multigoal_env import MultiGoalEnv
# from stac.envs.multigoal_env_obstacles import MultiGoalObstaclesEnv
# from stac.envs.multigoal_max_entropy_env import MultiGoalMaxEntropyEnv
# from stac.envs.multigoal_max_entropy_env_obstacles import MultiGoalMaxEntropyObstaclesEnv
import numpy as np
from datetime import datetime

from stac.svgd_factory import build_svgd_agent
from stac.utils import AttrDict
import timeit
from train import eval_call

CFGS = load_yaml(gp('rl/config.yaml'))
def get_arg_val(__args, __key):
    val = CFGS['common'][__key]
    if __args.task in CFGS.keys() and __key in CFGS[__args.task]:
        val = CFGS[__args.task][__key]
    return val

# TODO: Hyperparameter configurations

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu_id', type=int, default=0)
    # IMPORTANT: multigoal-max-entropy-obstacles and multigoal-obstacles should only be used at test time using a saved agent traned on the version of the environment without an obstacle.
    # parser.add_argument('--env', type=str, default='multigoal-max-entropy', choices=['Multigoal', 'multigoal-max-entropy', 'multigoal-max-entropy-obstacles', 'multigoal-obstacles', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'landmark', 'simple_landmark'])
    parser.add_argument('--task', type=str, default='Ant')
    parser.add_argument('--seed', '-s', type=int, default=42)
    # parser.add_argument('--actor', type=str, default='svgd_sql', choices=['svgd_sql', 'svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram'])
    parser.add_argument('--algo', type=str, choices=['SQL', 'SSAC'])

    ###### networks
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l_critic', type=int, default=2)
    parser.add_argument('--l_actor', type=int, default=2)
    parser.add_argument('--critic_activation', type=object, default=torch.nn.ReLU)
    parser.add_argument('--actor_activation', type=object, default=torch.nn.ReLU)

    ###### RL 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--a_c', type=float, default=0.2)
    parser.add_argument('--a_a', type=float, default=0.2)
    parser.add_argument('--replay_size', type=int, default=1e6)
    parser.add_argument('--load_replay', type=int, default=0)
    parser.add_argument('--max_experiment_steps', type=int, default=100_0000)
    parser.add_argument('--exploration_steps', type=int, default=0, help="pure exploration at the beginning of the training")
    parser.add_argument('--num_test_episodes', type=int, default=30)
    parser.add_argument('--num_final_test_episodes', type=int, default=100)
    parser.add_argument('--update_after', type=int, default=0)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--train_ratio', type=int, default=1)
    # parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--critic_cnn', action='store_true')
    ###### optim 
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    
    ###### action selection
    parser.add_argument('--train_action_selection', type=str, default='random', choices=['random', 'max', 'softmax', 'adaptive_softmax', 'softmax_egreedy'])
    parser.add_argument('--test_action_selection', type=str, default='random', choices=['random', 'max', 'softmax', 'adaptive_softmax', 'softmax_egreedy', 'amortized'])
    parser.add_argument('--svgd_particles', type=int, default=10)
    parser.add_argument('--svgd_steps', type=int, default=5)
    parser.add_argument('--svgd_lr', type=float, default=0.1)
    parser.add_argument('--svgd_sigma_p0', type=float, default=0.5)
    parser.add_argument('--svgd_kernel_sigma', type=float, default=None)
    parser.add_argument('--kernel_sigma_adaptive', type=int, default=4)
    parser.add_argument('--with_amor_infer', action='store_true')
   
    # tensorboard
    # parser.add_argument('--plot_format', type=str, default='pdf', choices=['png', 'jpeg', 'pdf', 'svg'])
    parser.add_argument('--stats_steps_freq', type=int, default=400) 
    parser.add_argument('--collect_stats_after', type=int, default=0)
    # parser.add_argument('--model_path', type=str, default='./evaluation_data/z_after/svgd_nonparam_999999')
    
    ###################################################################################
    # A label to differantiate experiments based on their importance (primary, secondary, and debugging experiments)
    parser.add_argument('--experiment_importance', type=str, default='dbg', choices=['prm', 'scn', 'dbg']) 
    # Load one checkpoint from a previous experiment, and evaluate the agent in that checkpoint
    # Evaluate each checkpoint of a previous experiment
    parser.add_argument('--all_checkpoints_test', type=int, default=0) 
    # Use the debugging hyperparameters (Used mainly to not edit the default parameters while debugging)
    parser.add_argument('--debugging', type=int, default=0)
    ###################################################################################
    # Adaptation args
    parser.add_argument("--amort_zdim", type=int, default=16)
    parser.add_argument("--apply_layer_norm", action='store_true')
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--sparse", type=bool, default=True)
    # parser.add_argument("--formal", action='store_true')
    parser.add_argument("--timer", action='store_true')
    parser.add_argument("--reward_type", type=str, default=CFGS['MultiGoalPointMaze']['reward_type'])
    parser.add_argument("--maze_map", type=str, default='simple')
    parser.add_argument("--smbgen_style", type=str, default='MultiFacet')
    parser.add_argument("--log_traj", action='store_true')
    parser.add_argument("--eval_points", type=int, default=CFGS['common']['eval_points'])
    parser.add_argument("--save_points", type=int, default=CFGS['common']['save_points'])
    parser.add_argument("--flag", type=str, default="informal")


    args = parser.parse_args()  
    args.debugging = bool(args.debugging)
    args.load_replay = bool(args.load_replay)
    args.all_check_points_test = bool(args.all_checkpoints_test)

    # NOTE: Update args using configs.yaml
    # args.max_experiment_steps = get_arg_val(args, 'steps')
    args.eval_points = get_arg_val(args, 'eval_points')
    args.num_test_episodes = get_arg_val(args, 'eval_episodes')
    args.num_final_test_episodes = get_arg_val(args, 'final_eval_episodes')

    # NOTE: Cover original actor selection with algo selection
    if args.algo == 'SQL':
        args.actor = 'svgd_sql'
        args.svgd_particles = 32
    elif args.algo == 'SSAC':
        args.actor = 'svgd_p0_pram'
    else:
        raise RuntimeError(f'Unsupported algortihm {args.algo}')

    ################# Configurations for a specific algorithm/environment #################
    if args.task in ['Hopper', 'Ant', 'Walker2d', 'Humanoid', 'HalfCheetah', 'Swimmer']:
        if args.task != 'Ant':
            args.a_a, args.a_c = 1.0, 1.0
        # args.svgd_steps = 3
        # args.l_critic = 3
        # args.l_actor = 3
        # args.actor_activation = torch.nn.GELU
        # args.critic_activation = torch.nn.GELU
        args.max_experiment_steps = get_arg_val(args, 'steps')
    if args.task == 'MultiGoalPointMaze':
        # if args.algo == 'SSAC':
        #     args.a_a, args.a_c = 0.6, 0.6
        args.num_test_episodes == CFGS['MultiGoalPointMaze']['eval_episodes']
        args.num_final_test_episodes = CFGS['MultiGoalPointMaze']['final_eval_episodes']
        key = f'pm_{args.maze_map}'
        args.max_experiment_steps = CFGS[key]['steps']
        if args.maze_map == 'hard':
            args.num_test_episodes = CFGS[key]['eval_episodes']
            args.num_final_test_episodes = CFGS[key]['final_eval_episodes']
        args.log_traj = True
    if args.task == 'MultiGoal':
        # args.max_experiment_steps = 20000
        args.max_experiment_steps = CFGS['MultiGoal']['steps']
        args.log_traj = True
    if args.task == 'MarioLevelGen':
        args.max_experiment_steps = 50_0000
        args.svgd_steps = 3

    if args.debugging:
        print('############################## DEBUGGING ###################################')
        args.exploration_steps = 1000
        args.max_experiment_steps = 30000
        args.num_test_episodes = 10
        print('############################################################################')

    # get device
    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    args.device =device
    # NOTE: New logging
    # if args.timer:
    #     root = 'timer'
    # elif args.formal:
    #     root = 'formal'
    # else:
    #     root = 'data'
    # if args.timer:
    #     args.save_points = 0
    #     args.eval_points = 0
    #     args.log_points = 100
    #     args.max_experiment_steps = 10_0000
    # elif args.formal:
    #     args.save_points = args.eval_points
    root = f'data/{args.flag}'
    if not args.path:
        # if args.timer:
        #     path = auto_dire(f'{root}/{args.algo}')
        # else:
        path = auto_dire(f'{root}/{args.task}/{args.algo}')
    else:
        path = gp(root, args.path)
        os.makedirs(path, exist_ok=True)
    if os.path.exists(f'{path}/final.pt'):
        print(f'Training at <{path}> is skipped as there has a finished trial already.')
        exit()
    print(f"Training data will be saved to {path}")
    if args.save_points > 0:
        os.makedirs(f'{path}/checkpoints', exist_ok=True)
    logger_keys = ['reward', 'eplen', 'reward-std', 'eplen-std']
    if 'MultiGoal' in args.task:
        logger_keys.append('reachable_modes')
        logger_keys.append('multi_goal_score')
    if 'MarioLevelGen' in args.task:
        logger_keys.append('avg-distance')
        logger_keys.append('gmean-distance')
    eval_logger = None if args.timer else CsvLogger(f'{path}/eval_log.csv', logger_keys, 0)
    if args.task == 'MultiGoalPointMaze':
        robustness_logger = CsvLogger(f'{path}/robustness.csv', ('removal-SR5', 'obstacle-SR5'), 0)
    else:
        robustness_logger = None
    if args.task == 'MarioLevelGen':
        os.makedirs(gp(path, 'generated_levels'), exist_ok=True)
    save_yaml({'start_time': now_datetime(), **vars(args)}, f'{path}/config.yaml')
    if args.log_traj:
        os.makedirs(f'{path}/trajectories', exist_ok=True)

    train_env = make_env(args)
    eval_env = make_env(args, True, 10)

    fix_seed(args.seed)
    train_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed)

    stac = build_svgd_agent(args, train_env, path, device)
    # Initial save and evaluation
    eval_env.reset(seed=args.seed)
    if 'MultiGoal' in args.task:
        os.makedirs(f'{path}/eval_trajs', exist_ok=True)
        trajs = rollout_pos_trajs(eval_env, stac, 50)
        viz_trajs(train_env, trajs, f'{path}/eval_trajs/step0', title=f'Behavior after initialization')
    if eval_logger is not None:
        if robustness_logger is not None:
            removal_SR5 = test_removal_robustness(args.maze_map, stac, train_env.unwrapped.num_goals // 2)
            obstacle_SR5 = test_eval_mode_robustness(args.maze_map, stac)
            robustness_logger.update(0, 0., 0., {'removal-SR5': removal_SR5, 'obstacle-SR5': obstacle_SR5})
        eval_call(eval_env, args.num_test_episodes, stac, eval_logger, (0, 0., 0.), path, args.seed)
        eval_itv = args.max_experiment_steps // args.eval_points
        eval_horizon = eval_itv
    if args.save_points > 0:
        torch.save(stac.ac.state_dict(), gp(f'{path}/checkpoints', f'0.pt'))
        # agent.save(gp(f'{path}/checkpoints', f'0.pt'))
        save_itv = args.max_experiment_steps // args.save_points
        save_horizon = save_itv
    else:
        save_itv, save_horizon = 2 * args.max_experiment_steps, 2 * args.max_experiment_steps
    # Training
    for info in stac.forward():
        # print(t, eval_horizon, time_elapsed)
        steps = info[0]
        if steps >= save_horizon:
            torch.save(stac.ac.state_dict(), gp(f'{path}/checkpoints', f'{steps}.pt'))
            # agent.save(gp(f'{path}/checkpoints', f'{steps}.pt'))
            save_horizon += save_itv
        if eval_logger is not None and steps >= eval_horizon:
            if robustness_logger is not None:
                removal_SR5 = test_removal_robustness(args.maze_map, stac, train_env.unwrapped.num_goals // 2)
                obstacle_SR5 = test_eval_mode_robustness(args.maze_map, stac)
                robustness_logger.update(*info, {'removal-SR5': removal_SR5, 'obstacle-SR5': obstacle_SR5})
            eval_call(eval_env, args.num_test_episodes, stac, eval_logger, info, path, args.seed)
            eval_horizon += eval_itv
            if 'MultiGoal' in args.task:
                trajs = rollout_pos_trajs(eval_env, stac, args.num_test_episodes)
                viz_trajs(train_env, trajs, f'{path}/eval_trajs/step{steps}', title=f'Behavior after {steps} steps')
    torch.save(stac.ac.state_dict(), f'{path}/final.pt')
    if not args.timer:
        results = eval_call(eval_env, args.num_final_test_episodes, stac, None, None, None, args.seed)
        if 'levels' in results.keys():
            results.pop('levels')
        save_json(results, f'{path}/final_scores.json')
    if eval_logger is not None:
        eval_logger.close(*info)
    if robustness_logger is not None:
        robustness_logger.close(*info)

    # stop = timeit.default_timer()
    # print('Time: ', stop - start)
    print('Experiment Finished.')
    # print(project_name)














