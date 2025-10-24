import gymnasium as gym
from envs.smbgen import VecMarioGenerationEnv


__REGISTERED = set()
__MUJOCO = {'Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Swimmer', 'Walker2d'}

def make_env(args, use_vec=False, vec_num=5, vec_mode='sync'):
    task = args.task
    env_kwargs = {}
    if 'Maze' in args.task:
        if 'Maze' not in __REGISTERED:
            from envs.mgmaze.point_maze import register_point_maze
            register_point_maze()
            from envs.mgmaze.ant_maze import register_ant_maze
            register_ant_maze()
            __REGISTERED.add('Maze')
        env_kwargs['reward_type'] = args.reward_type
        env_kwargs['maze_map'] = args.maze_map
    elif args.task == 'MultiGoal':
        if 'MultiGoal' not in __REGISTERED:
            from envs.customized import register_customized
            register_customized()
            __REGISTERED.add('MultiGoal')
        env_kwargs['sparse'] = args.sparse
    if 'MarioLevelGen' in args.task:
        from envs.smbgen import register_smbgen
        register_smbgen()
        env_kwargs['style'] = args.smbgen_style
        env_kwargs['device'] = args.device
    if task in __MUJOCO:
        task = task + '-v4'
    if 'Humanoid' in task:
        wrapper = lambda e: gym.wrappers.RescaleAction(e, -1.0, 1.0)
        if use_vec:
            return gym.make_vec(task, vec_num, vec_mode, wrappers=[wrapper], **env_kwargs)
        else:
            return wrapper(gym.make(task, **env_kwargs))
    if use_vec:
        if 'MarioLevelGen' in args.task:
            from envs.smbgen import VecMarioGenerationEnv
            return VecMarioGenerationEnv(num_envs=vec_num, **env_kwargs)
        return gym.make_vec(task, vec_num, vec_mode, **env_kwargs)
    else:
        return gym.make(task, **env_kwargs)

