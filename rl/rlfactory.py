import warnings
from argparse import Namespace
import itertools as itt
import numpy as np
import pandas
from myutils.filesys import load_yaml
from rl.agents import *
from rl.models.actors import *
from rl.models.critics import *
from rl.nets.policy import *
from rl.nets.vf import *
from rl.strategies import LinearScheduledCoefficient
from stac.svgd_factory import load_SVGD_agent


def build_agent(cfgs):
    # Configs
    if 'tar_entropy' in cfgs:
        if type(cfgs.tar_entropy) is str and cfgs.tar_entropy[-1] == 'd':
            cfgs.tar_entropy = float(cfgs.tar_entropy[:-1]) * cfgs.act_dim
        else:
            cfgs.tar_entropy= float(cfgs.tar_entropy)
    # Build nets
    cm_args = (cfgs.obs_dim, cfgs.act_dim)
    cm_kwargs = dict(
        hidden_size=cfgs.hidden_size, hidden_layers=cfgs.hidden_layers,
        layer_norm=cfgs.apply_layer_norm
    )
    match cfgs.algo:
        case "SAC":
            anet = ReparamGaussPolicyMLP(*cm_args, activation=cfgs.actor_activation, **cm_kwargs)
            cnet = DoubleQMLP(*cm_args, activation=cfgs.critic_activation, **cm_kwargs)
        case "DrAC" | "DrAC0":
            if cfgs.actor_type.lower() == 'diffusion':
                anet = MLPDiffusionPolicy(*cm_args, activation=cfgs.actor_activation, steps=cfgs.diffusion_steps, **cm_kwargs)
            else:
                anet = MLPLatentPolicy(*cm_args, z_dim=cfgs.z_dim, activation=cfgs.actor_activation, **cm_kwargs)
            cnet = DoubleQMLP(*cm_args, activation=cfgs.critic_activation, **cm_kwargs)
        case "DACER":
            anet = MLPDiffusionPolicy(*cm_args, activation=cfgs.actor_activation, steps=cfgs.diffusion_steps, **cm_kwargs)
            # DACER actor add noise and clip so do no clip at here
            cnet = DoubleGaussianMLP(*cm_args, activation=cfgs.critic_activation, **cm_kwargs)
        case _:
                raise NotImplementedError(f'Algorithm {cfgs.algo} is not supported')
    # Build models
    match cfgs.algo:
        case "SAC":
            actor = SoftActor(anet, lr=cfgs.actor_lr, tar_ent=cfgs.tar_entropy)
            critic = SoftDualClipCritic(cnet, gamma=cfgs.gamma, tau=cfgs.tau, lr=cfgs.critic_lr)
        case "DrAC0":
            warnings.warn('DrAC0 is deprecated', DeprecationWarning)
            if cfgs.start_alpha == cfgs.end_alpha:
                alpha = cfgs.end_alpha
            else:
                alpha = LinearScheduledCoefficient(cfgs.start_alpha, cfgs.end_alpha, cfgs.end_rate_alpha)
            actor = LatentActorV0(anet,
                lr=cfgs.actor_lr, alpha=alpha, reg_samples=cfgs.reg_samples)
            critic = LatentDualClipCritic(cnet, gamma=cfgs.gamma, tau=cfgs.tau, lr=cfgs.critic_lr)
        case "DrAC":
            beta = float(cfgs.beta)
            actor = LatentActor(
                anet, lr=cfgs.actor_lr, pg_samples=cfgs.pg_samples, reg_samples=cfgs.reg_samples,
                beta=beta)
            critic = LatentDualClipCritic(cnet, gamma=cfgs.gamma, tau=cfgs.tau, lr=cfgs.critic_lr)
        case "DACER":
            actor_kwargs = dict(
                lr=cfgs.actor_lr, tar_ent=cfgs.tar_entropy, alpha_lr=cfgs.alpha_lr, l=cfgs.l,
                ent_est_components=cfgs.ent_est_components, w0=cfgs.w0
            )
            actor = DACERActor(anet, tanh_out=cfgs.tanh_out, **actor_kwargs)
            critic = TriRefinedDistributionalCritic(cnet, gamma=cfgs.gamma, tau=cfgs.tau, lr=cfgs.critic_lr)
        case _:
            raise NotImplementedError(f'Algorithm {cfgs.algo} is not supported')
    # Build agent
    match cfgs.algo:
        case "SAC":
            agent = SAC(actor, critic, device=cfgs.device)
        case "DrAC0":
            agent = DrACV0(actor, critic, device=cfgs.device)
        case "DrAC":
            agent = DrAC(actor, critic, alpha_update_repeat=cfgs.alpha_update_repeat, device=cfgs.device)
        case "DACER":
            agent = DACER(actor, critic, cfgs.policy_delay, cfgs.alpha_update_itv, cfgs.device)
        case _:
            raise NotImplementedError(f'Algorithm {cfgs.algo} is not supported')
    return agent

def load_agent(folder, ckpt=None, device=None):    
    # TODO: Support with SVGD
    config = load_yaml(gp(folder, 'config.yaml'))
    if config['algo'] in ('SQL', 'SSAC'):
        return load_SVGD_agent(folder, ckpt, device)
    # Dealing with new configs not exist in old version
    defaults = load_yaml(gp('rl', 'config.yaml'))
    aux = [defaults[key].items() for key in ('common', config['algo'])]
    for k, v in itt.chain(*aux):
        if k not in config.keys():
            print(k, v)
            config[k] = v
    args = Namespace(**config)
    if device is not None: 
        args.device = device
    agent = build_agent(args)
    filepath = gp(folder, 'final.pt') if ckpt is None else gp(folder, 'checkpoints', f'{ckpt}.pt')
    agent.load(filepath, args.device)
    if args.algo == 'DACER':
        log = pandas.read_csv(gp(folder, 'agent_log.csv'))
        if ckpt is None:
            alpha = float(log['alpha'].to_numpy()[-1])
        else:
            steps = log['steps']
            alphas = log['alpha']
            i = np.argmin(np.abs(steps - int(ckpt)))
            alpha = float(alphas[i])
        agent.actor.alpha = alpha
    print(f'Loaded agent from {folder} to {device}')
    return agent, args
