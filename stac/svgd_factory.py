import os
from argparse import Namespace

import torch

from envs.factory import make_env
from myutils.filesys import load_yaml, gp
from rl.loggers import CsvLogger
from stac.actorcritic import ActorCritic
from stac.core import MaxEntrRL
from stac.utils import AttrDict


def build_svgd_agent(args, train_env, path, device, eval=False):
    # actor arguments
    if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
        actor_kwargs = AttrDict(num_svgd_particles=args.svgd_particles, num_svgd_steps=args.svgd_steps,
                                svgd_lr=args.svgd_lr, test_action_selection=args.test_action_selection,
                                svgd_sigma_p0=args.svgd_sigma_p0,
                                batch_size=args.batch_size, device=device, hidden_sizes=[args.hid] * args.l_actor,
                                activation=args.actor_activation, apply_layer_norm=args.apply_layer_norm,
                                kernel_sigma=args.svgd_kernel_sigma, adaptive_sig=args.kernel_sigma_adaptive,
                                alpha=args.a_a, with_amor_infer=args.with_amor_infer)

    elif args.actor == 'svgd_sql':
        actor_kwargs = AttrDict(num_svgd_particles=args.svgd_particles,
                                test_action_selection=args.test_action_selection, apply_layer_norm=args.apply_layer_norm,
                                batch_size=args.batch_size, device=device, hidden_sizes=[args.hid] * args.l_actor,
                                activation=args.actor_activation, kernel_sigma=args.svgd_kernel_sigma,
                                adaptive_sig=args.kernel_sigma_adaptive)
    elif args.actor == 'sac':
        actor_kwargs = AttrDict(hidden_sizes=[args.hid] * args.l_actor, apply_layer_norm=args.apply_layer_norm,
                                test_action_selection=args.test_action_selection,
                                device=device, activation=args.actor_activation, batch_size=args.batch_size)
    else:
        raise NotImplementedError(f'Unknown SVGD actor type: {args.actor}')
    critic_kwargs = AttrDict(hidden_sizes=[args.hid]*args.l_critic, activation=args.critic_activation, 
                             apply_layer_norm=args.apply_layer_norm, critic_cnn=args.critic_cnn)

    # RL args
    RL_kwargs = AttrDict(stats_steps_freq=args.stats_steps_freq, gamma=args.gamma,
        alpha_c=args.a_c, alpha_a=args.a_a, replay_size=int(args.replay_size), exploration_steps=args.exploration_steps,
        update_after=args.update_after, update_every=args.update_every,
        max_experiment_steps=int(args.max_experiment_steps), train_ratio=args.train_ratio,
        debugging=args.debugging,
        collect_stats_after=args.collect_stats_after, all_checkpoints_test=args.all_checkpoints_test,
        train_action_selection=args.train_action_selection)

    # optim args
    optim_kwargs = AttrDict(polyak=args.polyak,lr_critic=args.lr_critic, lr_actor=args.lr_actor,batch_size=args.batch_size)
    if not eval:
        # Logging the Hyperparameters used in the experiment
        print('########################################## Hyper-Parameters ##########################################')
        if not args.timer:
            print('Debugging: ', args.debugging)
            print('GPU ID: ', args.gpu_id)
            print('Environment: ', args.task)
            print('Algorithm: ', args.actor)
            print('Hidden layer size: ', args.hid)
            print('Critic\'s Number of layers: ', args.l_critic)
            if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                print('Actor\'s Number of layers: ', args.l_actor)
            print('Critic\'s Activation: ', args.critic_activation)
            if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                print('Actor\'s Activation: ', args.actor_activation)
            print('Discount Factor (Gamma): ', args.gamma)
            print('Entropy coefficient (Alpha Critic): ', args.a_c)
            print('Entropy coefficient (Alpha Actor): ', args.a_a)
            print('Replay Buffer size: ', args.replay_size)
            print('Load Replay Buffer: ', args.load_replay)
            print('Experiment\'s steps: ', args.max_experiment_steps)
            print('Initial Exploration steps: ', args.exploration_steps)
            print('Number test episodes: ', args.num_test_episodes)
            print('Start Updating models after step: ', args.update_after)
            print('Update models every: ', args.update_every)
            # print('Max Environment steps: ', args.max_steps)
            print('Polyak target update rate: ', args.polyak)
            print('Critic\'s learning rate: ', args.lr_critic)
            if args.actor not in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                print('Actor\'s learning rate: ', args.lr_actor)
            print('Batch size: ', args.batch_size)

            print('Train action selection: ', args.train_action_selection)
            print('Test action selection: ', args.test_action_selection)

            if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram', 'svgd_sql']:
                print('Number of particles for SVGD: ', args.svgd_particles)
                print('SVGD learning Rate: ', args.svgd_lr)
            if args.actor in ['svgd_nonparam', 'svgd_p0_pram', 'svgd_p0_kernel_pram']:
                print('Number of SVGD steps: ', args.svgd_steps)
                print('SVGD initial distribution\'s variance: ', args.svgd_sigma_p0)
                print('SVGD\'s kernel variance: ', args.svgd_kernel_sigma)
            # print('Plot format: ', args.plot_format)
            print('Statistics Collection frequency: ', args.stats_steps_freq)
            print('Collect Statistics after: ', args.collect_stats_after)
            print('Seed: ', args.seed)
            print('Device: ', device)
        # print('Project Name: ', project_name)
        print('Experiment Importance: ', args.experiment_importance)
        print('Experiment PID: ', os.getpid())
        print('######################################################################################################')


    log_traj = path if args.log_traj and not eval else None
    logger = None if eval else CsvLogger(f'{path}/agent_log.csv', ('loss_q', 'loss_pi'))
    stac=MaxEntrRL(train_env, task=args.task, actor=args.actor, device=device,
                   critic_kwargs=critic_kwargs, actor_kwargs=actor_kwargs,
                   RL_kwargs=RL_kwargs, optim_kwargs=optim_kwargs, logger=logger,
                   need_q=args.task in ["simple_landmark", "landmark"], log_traj=log_traj)
    return stac

def load_SVGD_agent(folder, ckpt=None, device='cuda:0'):
    args = Namespace(**load_yaml(gp(folder, "config.yaml")))
    args.actor_activation = torch.nn.ReLU
    args.critic_activation = torch.nn.ReLU
    args.log_traj = False
    args.device = device
    env = make_env(args)

    agent = build_svgd_agent(args, env, None, device, True)
    filepath = gp(folder, 'final.pt') if ckpt is None else gp(folder, 'checkpoints', f'{ckpt}.pt')
    agent.ac.load_state_dict(torch.load(filepath, map_location=device))
    return agent, args
