import matplotlib.pyplot as plt
from matplotlib.rcsetup import cycler
from analysis.collect import *


COLORS = ('#5047cd', '#aa3200', '#e8c85c', '#ae95ff', '#ff844f', '#007100', '#68baa5', '#98aae0', '#006789')
plt.rcParams['axes.prop_cycle'] = cycler(color=COLORS)
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['legend.framealpha'] = 0


def after_plot(xlabel='', ylabel='', title=None, fpath=None, legend=True, legend_kwargs={}, tight_pad=0.5):
    plt.suptitle(fpath if title is None else title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout(pad=tight_pad)
    if legend:
        plt.legend(**legend_kwargs)
    if fpath is None:
        plt.show()
    else:
        plt.savefig(gp(fpath))
    plt.clf()

def plot_single_trial(folder, logfile='eval_log.csv', xkey='steps', ykey='reward', xlabel=None, ylabel=None, title=None):
    x, y = read_xy_from_csv(gp(folder, logfile), xkey, ykey)
    plt.plot(x, y)
    xlabel = xkey.capitalize() if xlabel is None else xlabel
    ylabel = ykey.capitalize() if ylabel is None else ylabel
    after_plot(xlabel, ylabel, title, fpath=f'{folder}/{ylabel} by {xlabel}.png', legend=False)

def plot_line_X_trials(root, logfile='eval_log', xkey='steps', ykey='reward', horizon=None, ema=None, only_finished=True, line_label=None, line_kwargs={}, alpha=0.2, ax=None):
    x, ymean, ystd = read_meanstd_X_trials(root, xkey, ykey, logfile, horizon, ema, only_finished=only_finished)
    if x is None:
        return None
    if line_label is None: line_label = ykey
    if ax is None:
        line = plt.plot(x, ymean, label=line_label, **line_kwargs)
        color = line[0].get_color() if 'color' not in line_kwargs else line_kwargs['color']
        plt.fill_between(x, ymean-ystd, ymean+ystd, color=color, alpha=alpha, linewidth=0)
    else:
        line = ax.plot(x, ymean, label=line_label, **line_kwargs)
        color = line[0].get_color() if 'color' not in line_kwargs else line_kwargs['color']
        ax.fill_between(x, ymean-ystd, ymean+ystd, color=color, alpha=alpha, linewidth=0)
    return line[0]

def viz_trajs(env, trajs, path, title='Evaluation Trajectories'):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.subplots()
    env.unwrapped.plot(ax, 1)
    for traj in trajs:
        ax.plot(traj[:, 0], traj[:, 1], lw=1, alpha=0.5, color='magenta')
        # ax.scatter(traj[:, 0], traj[:, 1], marker='.', lw=0, s=24, color='magenta')
    final_x, final_y = [traj[-1, 0] for traj in trajs], [traj[-1, 1] for traj in trajs]
    ax.scatter(final_x, final_y, marker='x', s=32, color='navy', zorder=2)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    after_plot(title=title, fpath=path, legend=False)

def plot_discovered_modes(path, ax=None, label=None):
    x, y = read_discovered_modes(path)
    y_std = np.std(y, axis=0)
    y = np.mean(y, axis=0)
    if ax is None:
        plt.plot(x, y, label=label)
        plt.fill_between(x, y-y_std, y+y_std, alpha=0.2)
    else:
        ax.plot(x, y, label=label)
        ax.fill_between(x, y-y_std, y+y_std, alpha=0.2)
    pass

def plot_robustness_by_learning(path, fname, key, ax=None, label=None):
    x, y = get_robustness_by_learning(path, fname, key)
    y_std = np.std(y, axis=0)
    y = np.mean(y, axis=0)
    if ax is None:
        plt.plot(x, y, label=label)
        plt.fill_between(x, y-y_std, y+y_std, alpha=0.2)
    else:
        ax.plot(x, y, label=label)
        ax.fill_between(x, y-y_std, y+y_std, alpha=0.2)

def plot_robustness_by_learning(path, fname, key, ax=None, label=None):
    x, y = get_robustness_by_learning(path, fname, key)
    y_std = np.std(y, axis=0)
    y = np.mean(y, axis=0)
    if ax is None:
        plt.plot(x, y, label=label)
        plt.fill_between(x, y-y_std, y+y_std, alpha=0.2)
    else:
        ax.plot(x, y, label=label)
        ax.fill_between(x, y-y_std, y+y_std, alpha=0.2)

