import argparse

from myutils.img import make_img_sheet
from myutils.tools import fix_seed
from rl.rlfactory import build_agent
from analysis.viz import plot_single_trial, viz_trajs
from myutils.filesys import *
from myutils.datastruct import recursive_update
from myutils.fmt import now_datetime
from envs.factory import make_env
from analysis.test import *
from rl.loggers import *
from rl.repmem import *
from rl.trainers import OffPolicyTrainer


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return list(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return list(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def setup(parser, algo):
    parser.add_argument("--configs", nargs="+", default=[])
    parser.add_argument("--task", type=str)
    args, _ = parser.parse_known_args()
    configs = load_yaml("rl/config.yaml")
    defaults = {}
    for name in ["common", algo]:
        recursive_update(defaults, configs[name])
    if args.task in configs.keys():
        recursive_update(defaults, configs[args.task])
    if args.task in configs[algo].keys():
        recursive_update(defaults, configs[algo][args.task])
    for name in args.configs:
        recursive_update(defaults, configs[name])
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        tp = args_type(value)
        if value is False:
            parser.add_argument(f"--{key}", action='store_true')
        else:
            parser.add_argument(f"--{key}", type=tp, default=tp(value))

    parser.add_argument("--path", type=str, default='')
    return parser


__ALGOS = ('SAC', 'DrAC', 'DACER') # SQL and S^2AC are in train_svgd.py

def eval_call(eval_envs, eval_eps, agent, logger, logger_info, path, seed):
    results = std_rl_test(eval_envs, agent, eval_eps, seed)
    if logger is not None and 'avg-distance' in logger.keys:
        results.update(smbgen_diversity_test(eval_envs, agent, seed=seed))
    if logger is not None:
        logger.update(*logger_info, results)
        if 'levels' in results.keys():
            imgs = [lvl[:, :160].to_img() for lvl in results['levels'][:10]]
            make_img_sheet(imgs, 1, save_path=gp(path, 'generated_levels', f'{logger_info[0]}.png'))
        if logger_info[0] == 0:
            return
        title = path[len(PRJROOT):]
        plot_single_trial(path, title=title)
        if 'reachable_modes' in logger.keys:
            plot_single_trial(path, ykey='reachable_modes', ylabel='Reachable_modes', title=title)
            plot_single_trial(path, ykey='multi_goal_score', ylabel='Multi-goal score', title=title)
        if os.path.exists(gp(path, 'robustness.csv')):
            plot_single_trial(path, 'robustness.csv', ykey='removal-SR5', ylabel='SR5-Removal', title=title)
            plot_single_trial(path, 'robustness.csv', ykey='obstacle-SR5', ylabel='SR5-Obstacle', title=title)
        if 'levels' in results.keys():
            plot_single_trial(path, ykey='gmean-distance', ylabel='Diversity', title=title)
    else:
        return results

def build_up(args, obs_space, act_space):
    # Check valid
    match args.algo:
        case "SAC":
            assert (act_space.low == -1).all() and (act_space.high == 1).all(), \
                "action space must be bounded in [-1, 1]"

    if args.timer:
        root = 'timer'
    # elif args.formal:
    #     root = 'formal'
    # else:
    #     root = 'data'
    root = f'data/{args.flag}'
    # if args.timer:
    #     args.save_points = 0
    #     args.eval_points = 0
    #     args.log_points = 100
    #     args.steps = 10_0000
    # elif args.formal:
    # args.save_points = args.eval_points
    if not args.path:
        if args.timer:
            path = auto_dire(f'{root}/{args.algo}')
        else:
            path = auto_dire(f'{root}/{args.task}/{args.algo}')
    else:
        path = gp(root, args.path)
        os.makedirs(path, exist_ok=True)
    if os.path.exists(f'{path}/final_scores.json'):
        print(f'Training at <{path}> is skipped as there has a finished trial already.')
        exit()
    if args.save_points > 0:
        os.makedirs(f'{path}/checkpoints', exist_ok=True)
    args.obs_dim = obs_space.shape[0]
    args.act_dim = act_space.shape[0]
    agent = build_agent(args)
    logger_keys = ['reward', 'eplen', 'reward-std', 'eplen-std']
    if 'MultiGoal' in args.task:
        logger_keys.append('reachable_modes')
        logger_keys.append('multi_goal_score')
    if 'MarioLevelGen' in args.task:
        logger_keys.append('avg-distance')
        logger_keys.append('gmean-distance')
    eval_logger = None if args.timer else CsvLogger(f'{path}/eval_log.csv', logger_keys, 0)

    if args.learning_start is None: 
        args.learning_start = args.batch_size
    trainer_kwargs = dict(
        self_expl_start=args.self_expl_start, learning_start=args.learning_start, train_ratio=args.train_ratio, 
        batch_size=args.batch_size, reward_scale=args.reward_scale
    )

    capacity = min(args.buffer_size, args.steps)
    if args.apply_per:
        rep_mem = PERMem(capacity, args.per_alpha, args.per_beta_start, args.per_beta_final, device=args.device)
    else:
        rep_mem = ReplayMem(args.obs_dim, args.act_dim, capacity, device=args.device)

    trainer = OffPolicyTrainer(rep_mem, **trainer_kwargs)
    return agent, trainer, eval_logger, path

def main(args):
    args.device = 'cpu' if args.gpuid < 0 else f'cuda:{args.gpuid}'
    # Make envs
    env = make_env(args)
    eval_envs = make_env(args, True, args.eval_envs)
    # Set seeds
    if args.seed is not None:
        args.seed = int(args.seed)
        fix_seed(args.seed)
        env.reset(seed=args.seed)
        eval_envs.reset(seed=args.seed)

    agent, trainer, eval_logger, path = build_up(args, env.observation_space, env.action_space)
    save_yaml({'start_time': now_datetime(), **vars(args)}, f'{path}/config.yaml')
    if 'MultiGoal' in args.task:
        os.makedirs(f'{path}/eval_trajs', exist_ok=True)
        trajs = rollout_pos_trajs(eval_envs, agent, args.eval_episodes)
        viz_trajs(env, trajs, f'{path}/eval_trajs/step0', title=f'Behavior after initialization')
    # Initial save and evaluation
    if args.task == 'MultiGoalPointMaze':
        robustness_logger = CsvLogger(f'{path}/robustness.csv', ('removal-SR5', 'obstacle-SR5'), 0)
    else:
        robustness_logger = None
    if args.task == 'MarioLevelGen':
        os.makedirs(gp(path, 'generated_levels'), exist_ok=True)
    if args.eval_points > 0:
        if robustness_logger is not None:
            removal_SR5 = test_removal_robustness(args.maze_map, agent, env.unwrapped.num_goals // 2)
            obstacle_SR5 = test_eval_mode_robustness(args.maze_map, agent)
            robustness_logger.update(0, 0., 0., {'removal-SR5': removal_SR5, 'obstacle-SR5': obstacle_SR5})
        eval_call(eval_envs, args.eval_episodes, agent, eval_logger, (0, 0., 0.), path, args.seed)
        eval_itv = args.steps // args.eval_points
        eval_horizon = eval_itv
    if args.save_points > 0:
        agent.save(gp(f'{path}/checkpoints', f'0.pt'))
        save_itv = args.steps // args.save_points
        save_horizon = save_itv
    else:
        save_itv, save_horizon = 2 * args.steps, 2 * args.steps
    train_args = (env, agent, args.steps, path, args.log_points, args.log_trajs)
    # Training
    for info in trainer.train(*train_args):
        steps = info[0]
        if steps >= save_horizon:
            agent.save(gp(f'{path}/checkpoints', f'{steps}.pt'))
            save_horizon += save_itv
        if args.eval_points > 0 and steps >= eval_horizon:
            if robustness_logger is not None:
                removal_SR5 = test_removal_robustness(args.maze_map, agent, env.unwrapped.num_goals // 2)
                obstacle_SR5 = test_eval_mode_robustness(args.maze_map, agent)
                robustness_logger.update(*info, {'removal-SR5': removal_SR5, 'obstacle-SR5': obstacle_SR5})
            eval_call(eval_envs, args.eval_episodes, agent, eval_logger, info, path, args.seed)
            eval_horizon += eval_itv
            if 'MultiGoal' in args.task:
                trajs = rollout_pos_trajs(eval_envs, agent, args.eval_episodes)
                viz_trajs(env, trajs, f'{path}/eval_trajs/step{steps}', title=f'Behavior after {steps} steps')
    if not args.timer:
        results = eval_call(eval_envs, args.final_eval_episodes, agent, None, None, None, args.seed)
        if 'levels' in results.keys():
            results.pop('levels')
        save_json(results, f'{path}/final_scores.json')

    eval_envs.close()
    if eval_logger is not None:
        eval_logger.close(*info)
    if robustness_logger is not None:
        robustness_logger.close(*info)
    env.close()

# Reduce robustness eval repeats to 50
# Deal with config
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    for algo in __ALGOS:
        subparser = subparsers.add_parser(algo, help=f'Train {algo}')
        subparser.set_defaults(algo=algo)
        subparser = setup(subparser, algo)    

    args = parser.parse_args()
    main(args)


