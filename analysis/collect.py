import csv
import os
import glob
import numpy as np
import pandas as pd
import itertools as itt
from argparse import Namespace
from envs.factory import make_env
from analysis.tools import ema_smoothing
from myutils.filesys import gp, load_json, load_yaml
from myutils.mathtools import hv2d


def read_xy_from_csv(filepath, xkey, ykey):
    df = pd.read_csv(gp(filepath))
    return df[xkey].values, df[ykey].values

def read_meanstd_X_trials(root, xkey='steps', ykey='reward', logname='eval_log', horizon=None, ema=None, only_finished=True):
    ys = []
    x = None
    for folder in glob.glob(gp(root, 'trial*')):
        finised = os.path.exists(gp(folder, 'final.pt'))
        if only_finished and not finised:
            continue
        if not os.path.exists(gp(folder, f'{logname}.csv')):
            continue
        df = pd.read_csv(gp(folder, f'{logname}.csv'))
        y = df[ykey].values
        _x = df[xkey].values
        if horizon is not None:
            end = 0
            while end < len(_x) and _x[end] <= horizon:
                end += 1
            _x = _x[:end]
        if x is None or len(_x) < len(x):
            x = _x
        if ema is not None:
            y = ema_smoothing(y, alpha=ema)
        ys.append(y)
    try:
        y = np.stack([item[:len(x)] for item in ys])
        ymean = np.mean(y, 0)
        ystd = np.std(y, 0)
        return x, ymean, ystd
    except ValueError:
        print('No data found at', root)
        return None, None, None

def get_final_scores(root, key, logname='final_scores'):
    scores = []
    for folder in glob.glob(gp(root, 'trial*')):
        finised = os.path.exists(gp(folder, 'final_scores.json'))
        if not finised:
            print(f'No {logname}.json file at:', root)
            continue
        data = load_json(gp(folder, f'{logname}.json'))
        scores.append(data[key])
    return scores

def get_modes_freqs(folder, itv=500):
    config = load_yaml(gp(folder, 'config.yaml'))
    defaults = load_yaml(gp('rl', 'config.yaml'))
    aux = [defaults[key].items() for key in ('common', config['algo'])]
    for k, v in itt.chain(*aux):
        if k not in config.keys():
            print(k, v)
            config[k] = v
    args = Namespace(**config)
    env = make_env(args).unwrapped
    modes = np.zeros([env.num_goals + 1], dtype=int)
    steps = []
    modes_curve = []
    end_points = np.load(f'{folder}/trajectories/end_points.npy')
    final_states = np.load(f'{folder}/trajectories/final_states.npy')

    horizon = itv
    f = open(gp(folder, 'discovered_modes.csv'), 'w')
    writer = csv.writer(f)
    writer.writerow(['step'] + [f'mode{i}' for i in range(env.num_goals + 1)])
    for t, s in zip(end_points, final_states):
        while t >= horizon and horizon != end_points[-1]:    
            modes_curve.append(modes.copy())
            row = [str(horizon), *map(str, modes)]
            writer.writerow(row)
            steps.append(horizon)
            horizon += itv
        _, mode = env.compute_terminated(s[:2])
        modes[mode] += 1
    modes_curve.append(modes.copy())
    row = [str(horizon), *map(str, modes)]
    writer.writerow(row)
    steps.append(horizon)
    f.close()
    env.close()

def read_discovered_modes(path):
    x = None
    y = []
    for folder in glob.glob(gp(path, 'trial*')):
        data = pd.read_csv(f'{folder}/discovered_modes.csv')
        num_goals = len(data.columns) - 1
        discovered = []
        if x is None:
            x = data['step'].to_numpy()
        for _, row in data.iterrows():
            cur = 0
            for i in range(1, num_goals):
                if row[f'mode{i}'] > 0:
                    cur += 1
            discovered.append(cur)
        y.append(discovered)
    y = np.array(y)
    return x, y

def read_first_time_discovered_all(path, num_goals):
    x, y = read_discovered_modes(path)
    res = []
    for trial in y:
        for t, y in zip(x, trial):
            if y == num_goals:
                res.append(t)
    return res

def get_robustness_by_learning(path, ptb_type, key):
    x = None
    y = []
    for folder in glob.glob(gp(path, 'trial*')):
        fpath = f'{folder}/robustness/by_step_{ptb_type}.csv'
        if not os.path.exists(fpath):
            continue
        data = pd.read_csv(fpath)
        if x is None:
            x = data['step'].to_numpy()
        y.append(data[key].to_numpy())
    if not y:
        print(folder)
    return x, np.stack(y)

def read_unit_time(algo):
    unit_time, n = 0, 0
    for path in glob.glob(gp('timer', algo, 'trial*')):
        data = pd.read_csv(f'{path}/agent_log.csv')
        steps = data['steps'].to_numpy()[-1]
        total_time = data['time'].to_numpy()[-1]
        unit_time += total_time / steps * 1000
        n += 1
    return unit_time / n

def read_final_performance(folder, logfile='final_scores.json', key='reward'):
    vals = []
    for path in glob.glob(gp(folder, 'trial*')):
        if not os.path.exists(f'{path}/final.pt'):
            return
        ext_name = logfile.split('.')[-1]
        if ext_name == 'json':
            data = load_json(f'{path}/{logfile}')        
            vals.append(data[key])
        elif ext_name == 'csv':
            data = pd.read_csv(f'{path}/{logfile}')
            vals.append(data[key].to_numpy()[-1])
    return np.mean(vals), np.std(vals)

def get_smbgen_qd_coord(style, algo):
    q = {}
    d = {}
    for temp in ('low', 'mid', 'high'):
        x, y = [], []
        for folder in glob.glob(gp(f'formal/smbgen/{style}/{temp}/{algo}/trial*')):
            if not os.path.exists(gp(folder, 'final.pt')):
                continue
            data = pd.read_csv(gp(f'{folder}/eval_log.csv'))
            x.append(data['avg-distance'].to_numpy()[-1])
            y.append(data['reward'].to_numpy()[-1])
        q[temp] = np.mean(x)
        d[temp] = np.mean(y)
    return q, d

def get_smbgen_hv(style, algo):
    seed = 0
    temps = ('low', 'mid', 'high')
    root = f'formal/smbgen/{style}'
    steps = None
    hvs = []
    while True:
        if any([not os.path.exists(gp(f'{root}/{temp}/{algo}/trial{seed}', 'final.pt')) for temp in temps]):
            break
        xs = {}
        ys = {}
        hv_traj = []
        for temp in temps:
            data = pd.read_csv(gp(f'{root}/{temp}/{algo}/trial{seed}/eval_log.csv'))
            if steps is None:
                steps = data['steps'].to_numpy()
            xs[temp] = data['avg-distance'].to_numpy()
            ys[temp] = data['reward'].to_numpy()
        for i in range(len(steps)):
            locations = [(xs[temp][i], ys[temp][i]) for temp in temps]
            hv_traj.append(hv2d(locations))
        hvs.append(hv_traj)
        seed += 1
    hvs = np.array(hvs)
    return steps, np.mean(hvs, axis=0), np.std(hvs, axis=0)


if __name__ == '__main__':
    get_smbgen_hv('MarioPuzzle', 'DrAmt')
