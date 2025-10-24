import matplotlib.pyplot as plt
from analysis.make import *


def make_mgmaze_ablation():
    levels = ('simple', 'medium', 'hard')
    metrics = ('reward', 'reachable_modes')
    ylabels = ('Total reward', 'Reachable modes')
    algos = {
        'DrAmt': {'name': 'DrAmort', 't_name': '$\\beta$', 't_label': 'beta', 'temps': ('0.6', '0.7', '0.8', '0.9')},
        'DrDfs': {'name': 'DrDiffus', 't_name': '$\\beta$', 't_label': 'beta', 'temps': ('0.6', '0.7', '0.8', '0.9')},
        'DACER-tanh': {'name': 'DACER', 't_name': '$\\hat \\mathcal{H}$', 't_label': 'h', 'temps': ('0.8', '1.0', '1.2')},
        'SAC': {'name': 'SAC', 't_name': '$\\hat \\mathcal{H}$', 't_label': 'h', 'temps': ('0.8', '1.0', '1.2')},
        'SQL': {'name': 'SQL', 't_name': '$\\alpha$', 't_label': 'alpha', 'temps': ('0.2', '0.3', '0.4')},
        'SSAC': {'name': 'S$^2$AC', 't_name': '$\\alpha$', 't_label': 'alpha', 'temps': ('0.6', '0.7', '0.8')}
    }
    for k, info in algos.items():
        _, axs = plt.subplots(2, 3, figsize=(8, 5))
        name, t_name, t_label, temps = info['name'], info['t_name'], info['t_label'], info['temps']
        os.makedirs(f'data/ablation/PointMaze/figs/{k}', exist_ok=True)
        i = 0
        for metric, ylabel in zip(metrics, ylabels):
            row = axs[i]
            i += 1
            for level, ax in zip(levels, row):
                for temp in temps:
                    root = f'data/ablation/PointMaze/{level}/{k}/{t_label}{temp}'
                    plot_line_X_trials(root, line_label=f'{t_name}={temp}', ykey=metric, ax=ax, ema=0.5)
                if metric == 'reachable_modes':
                    ax.set_ylim(0, 8 if level == 'hard' else 4)
                else:
                    ax.set_ylim(0, 105)
                    ax.set_yticks(np.linspace(0, 100, 6), [*map(lambda x: '%.1f' % x, np.linspace(0, 1.0, 6))])
                ax.set_title(level.capitalize())
                ax.set_xlabel('Training steps')
                ax.ticklabel_format(scilimits=(-4, 4), axis='x')
                ax.legend(
                    fontsize=8, ncol=2, borderaxespad=0.2, borderpad=0.2,
                    handletextpad=0.3, columnspacing=0.5, handlelength=1.5
                )
                ax.set_ylabel(ylabel)
        fname = k.replace('-tanh', '') if k == 'DACER-tanh' else k
        after_plot(
            title=f'Learning curves of {name} with varied temperatures', fpath=f'figures/PM/ablation_{fname}.png',
            legend_kwargs={'fontsize': 8, 'ncols': 2}, tight_pad=1.0
        )

def make_mgmaze_comparison():
    levels = ('simple', 'medium', 'hard')
    metrics = ('reward', 'reachable_modes', 'removal-SR5', 'obstacle-SR5')
    ylabels = ('Total reward', 'Reachable modes', 'Robustness-Removal', 'Robustness-Obstacle')
    algos = {
        'DrAmt/beta0.8': 'DrAmort',
        'SQL/alpha0.3': 'SQL',
        'DrDfs/beta0.8': 'DrDiffus',
        'DACER-tanh/h1.0': 'DACER',
        'SAC/h1.0': 'SAC',
        'SSAC/alpha0.7': 'S$^2$AC',
    }
    os.makedirs(f'data/comparison/PointMaze/figs', exist_ok=True)
    for metric, ylabel in zip(metrics, ylabels):
        _, axs = plt.subplots(1, 3, figsize=(9, 3))
        for level, ax in zip(levels, axs):
            for fd, name in algos.items():
                root = f'data/ablation/PointMaze/{level}/{fd}'
                logfile = 'robustness' if 'SR5' in metric else 'eval_log'
                plot_line_X_trials(root, logfile, line_label=name, ykey=metric, ax=ax, ema=0.5)
            if metric == 'reachable_modes':
                ax.set_ylim(0, 8 if level == 'hard' else 4)
            elif metric == 'reward':
                ax.set_ylim(0, 105)
                ax.set_yticks(np.linspace(0, 100, 6), [*map(lambda x: '%.1f' % x, np.linspace(0, 1.0, 6))])
            else:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.linspace(0, 1.0, 6), [*map(lambda x: '%.1f' % x, np.linspace(0, 1.0, 6))])
            ax.set_title(level.capitalize())
            ax.set_xlabel('Training steps')
            ax.ticklabel_format(scilimits=(-4, 4), axis='x')
            ax.legend(
                fontsize=8, ncol=2, borderaxespad=0.2, borderpad=0.2,
                handletextpad=0.3, columnspacing=0.5, handlelength=1.5
            )
        fname = ylabel.replace(' ', '-')
        after_plot(
            fpath=f'figures/PM/{fname}.png', ylabel=ylabel, xlabel='Training steps',
            legend_kwargs={'fontsize': 8, 'ncols': 2}, title=''
        )


def make_mujoco_ablation():
    tasks = ('Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Swimmer', 'Walker2d')
    algos = {
        'DrAmt': {'t_name': '$\\beta$', 't_label': 'beta', 'temps': ('0.2', '0.3', '0.4', '0.5'), 'name': 'DrAmort'},
        'DrDfs': {'t_name': '$\\beta$', 't_label': 'beta', 'temps': ('0.2', '0.3', '0.4', '0.5'), 'name': 'DrDiffus'},
    }
    os.makedirs(f'data/ablation/mujoco/figs', exist_ok=True)
    for algo, info in algos.items():
        _, axs = plt.subplots(2, 3, figsize=(8, 5))
        t_name, t_label, temps = info['t_name'], info['t_label'], info['temps']
        for i, task in enumerate(tasks):
            ax = axs[i//3][i%3]
            for temp in temps:
                root = f'data/ablation/mujoco/{task}/{algo}/{t_label}{temp}'
                plot_line_X_trials(root, line_label=f'{t_name}={temp}', ax=ax)
            # SAC as baseline
            # root = f'data/ablation/mujoco/{algo}/{t_label}{temp}'
            # plot_line_X_trials(root, line_label=f'{t_name}={temp}', ax=ax)
            ax.set_title(f'{task}-v4')
            ax.set_xlabel('Training steps')
            # ax.ticklabel_format(scilimits=(-4, 4), axis='x')
            ax.legend(
                fontsize=8, ncol=2, borderaxespad=0.2, borderpad=0.2, loc='lower right',
                handletextpad=0.3, columnspacing=0.5, handlelength=1.5
            )

        after_plot(
            fpath=f'figures/mujoco/ablation-{algo}.png', title=info['name'],
            legend_kwargs={'fontsize': 8, 'ncols': 2, 'loc': 'lower right'}
        )

def make_mujoco_comparison():
    tasks = ('Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Swimmer', 'Walker2d')
    algos = {
        'ablation/mujoco/%s/DrAmt/beta0.5': 'DrAmort',
        'ablation/mujoco/%s/DrDfs/beta0.2': 'DrDiffus',
        'comparison/mujoco/%s/DACER': 'DACER',
        'comparison/mujoco/%s/SQL': 'SQL',
        'comparison/mujoco/%s/SAC': 'SAC',
    }
    os.makedirs(f'data/comparison/mujoco/figs', exist_ok=True)
    _, axs = plt.subplots(2, 3, figsize=(9, 5))
    for i, task in enumerate(tasks):
        ax = axs[i//3][i%3]
        for fd, name in algos.items():
            root = 'data/' + (fd % task)
            plot_line_X_trials(root, line_label=name, ax=ax)

        ax.set_title(task)
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Episodic return ')
        ax.ticklabel_format(scilimits=(-4, 4), axis='x')
        ax.tick_params(axis='y', labelrotation=60, labelsize=8)
        ax.legend(
            fontsize=8, ncol=2, borderaxespad=0.2, borderpad=0.2,
            handletextpad=0.3, columnspacing=0.5, handlelength=1.5
        )
    after_plot(
        'Training steps', 'Episodic return', fpath=f'figures/mujoco/mujoco-comparison.png', tight_pad=1.0,
        legend_kwargs={'fontsize': 8, 'ncols': 2}, title=''
    )

def __make_mgmaze_dacer_itv():
    levels = ('simple', 'medium', 'hard')
    metrics = ('reward', 'reachable_modes', 'removal-SR5', 'obstacle-SR5')
    ylabels = ('Total reward', 'Reachable modes', 'Robustness-Removal', 'Robustness-Obstacle')
    # algos = {
    #     'DrAmt': {'name': 'DrAmort', 't_name': '$\\beta$', 't_label': 'beta', 'temps': ('0.5', '0.6', '0.7', '0.8', '0.9')},
    #     'DrDfs': {'name': 'DrDiffus', 't_name': '$\\beta$', 't_label': 'beta', 'temps': ('0.5', '0.6', '0.7', '0.8', '0.9')},
    #     'DACER-tanh': {'name': 'DACER', 't_name': '$\\hat \\mathcal{H}$', 't_label': 'h', 'temps': ('0.8', '1.0', '1.2')},
    #     'SAC': {'name': 'SAC', 't_name': '$\\hat \\mathcal{H}$', 't_label': 'h', 'temps': ('0.8', '1.0', '1.2')},
    #     'SQL': {'name': 'SQL', 't_name': '$\\alpha$', 't_label': 'alpha', 'temps': ('0.3', '0.4', '0.5', '0.6')},
    #     'SSAC': {'name': 'S$^2$AC', 't_name': '$\\alpha$', 't_label': 'alpha', 'temps': ('0.6', '0.7')}
    # }
    for metric, ylabel in zip(metrics, ylabels):
        for itv in (250, 300, 500):
            _, axs = plt.subplots(1, 3, figsize=(9, 3))
            name, t_name = 'DACER', 'itv'
            os.makedirs(f'data/ablation/PointMaze/figs/DACER/itv{itv}', exist_ok=True)
            for level, ax in zip(levels, axs):
                for h in ('0.8', '1.0', '1.2'):
                    root = f'data/ablation/PointMaze/{level}/DACER-tanh-itv{itv}/h{h}'
                    logfile = 'robustness' if 'SR5' in metric else 'eval_log' 
                    plot_line_X_trials(root, logfile, line_label=f'{t_name}={itv}', ykey=metric, ax=ax, ema=0.3)
                if metric == 'reachable_modes':
                    ax.set_ylim(0, 8 if level == 'hard' else 4)
                elif metric == 'reward':
                    ax.set_ylim(0, 105)
                    ax.set_yticks(np.linspace(0, 100, 6), [*map(lambda x: '%.1f' % x, np.linspace(0, 1.0, 6))])
                else:
                    ax.set_ylim(0, 1)
                ax.set_title(level.capitalize())
                ax.set_xlabel('Training steps')
                ax.ticklabel_format(scilimits=(-4, 4), axis='x')
                ax.legend(
                    fontsize=8, ncol=2, borderaxespad=0.2, borderpad=0.2,
                    handletextpad=0.3, columnspacing=0.5, handlelength=1.5
                )

            after_plot(
                title=f'{name}: {ylabel}', fpath=f'data/ablation/PointMaze/figs/DACER/itv{itv}/{ylabel}.png', ylabel=ylabel, 
                legend_kwargs={'fontsize': 8, 'ncols': 2}
            )


# def make_smbgen_qd():
#     _, axs = plt.subplots(1, 2, figsize=(6, 2.75))
#     styles = ('MarioPuzzle', 'MultiFacet')
#     legend_kwargs = dict(
#         fontsize=8, ncol=3, loc='lower center', borderaxespad=0.2, handlelength=0.8,
#         borderpad=0.2, handletextpad=0.2, columnspacing=0.8
#     )
#     xlims = ((10, 40), (25, 50))
#     ylims = ((300, 1800), (300, 1050))
#     for style, ax, xlim, ylim in zip(styles, axs, xlims, ylims):
#         for algo, name, c in zip(algos, algo_names, COLORS):
#             q, d = get_smbgen_qd_coord(style, algo)
#             ax.scatter(q, d, color=c, label=name, marker='x', lw=1, s=20)
#         ax.set_title(style)
#         ax.set_xticks(np.linspace(*xlim, 6))
#         ax.set_yticks(np.linspace(*ylim, 6))
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
#         ax.set_xlabel('Episodic Return')
#         ax.set_ylabel('Diversity')
#         ax.legend(**legend_kwargs)
#     after_plot('Episodic Return', 'Diversity', style, fpath=f'formal/smbgen/QD-loc.png', legend_kwargs=legend_kwargs)


# def make_smb_ablation():
#     styles = ('MarioPuzzle', 'MultiFacet')
#     xlims = ((0, 45), (25, 50))
#     ylims = ((0, 1800), (0, 1200))
#     algos = {
#         'DrAmt': {'name': 'DrAmort', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
#         'DrDfs': {'name': 'DrDiffus', 'dires': ('beta0.2', 'beta0.5', 'beta0.8'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
#         'DACER': {'name': 'DACER', 'dires': ('low', 'mid', 'high'), 'labels': {r'$\hat H=-|\mathcal{A}|$', r'$\hat H=-0.25|\mathcal{A}|$', r'$\hat H=0.5|\mathcal{A}|$'}},
#         'SAC': {'name': 'SAC', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-|\mathcal{A}|$', r'$\hat H=-0.25|\mathcal{A}|$', r'$\hat H=0.5|\mathcal{A}|$')},
#         'SQL': {'name': 'SQL', 'dires': ('alpha0.05', 'alpha0.175', 'alpha0.3'), 'labels': (r'$\alpha=0.05$', r'$\alpha=0.175$', r'$\alpha=0.3$')},
#         'SSAC': {'name': 'S$^2$AC', 'dires': ('alpha0.1', 'alpha0.4', 'alpha0.7'), 'labels': (r'$\alpha=0.1$', r'$\alpha=0.4$', r'$\alpha=0.7$')}
#     }
#     legend_kwargs = dict(
#         loc='lower left', borderaxespad=0.2, handlelength=0.8,
#         borderpad=0.2, handletextpad=0.2, columnspacing=0.8
#     )
#     def __collect(__folder):
#         q = []
#         d = []
#         for folder in glob.glob(gp(__folder, '*')):
#             if not os.path.exists(gp(folder, 'final.pt')):
#                 continue
#             data = pd.read_csv(gp(f'{folder}/eval_log.csv'))
#             q.append(data['reward'].to_numpy()[-1])
#             d.append(data['avg-distance'].to_numpy()[-1])
#         # q[temp] = np.mean(x)
#         # d[temp] = np.mean(y)
#         # print(__folder)
#         # print(q)
#         # print(d)    
#         return q, d

#     for style, xlim, ylim in zip(styles, xlims, ylims):
#         _, axs = plt.subplots(2, 3, figsize=(10, 6))
#         ax_iter = itt.chain(axs[0], axs[1])
#         for k, info in algos.items():
#             ax = next(ax_iter)
#             for dire, label in zip(info['dires'], info['labels']):
#                 q, d = __collect(f'data/ablation/smbgen/{style}/{k}/{dire}')
#                 ax.scatter(q, d, label=label, marker='x', lw=1, s=20)
#                 ax.legend(**legend_kwargs)
#                 ax.set_title(info['name'])
#                 ax.set_xlim(xlim)
#                 ax.set_ylim(ylim)
#                 ax.set_xlabel('Quality')
#                 ax.set_ylabel('Diversity')
#                 # os.makedirs(f'data/ablation/PointMaze/figs/{k}', exist_ok=True)
#             # i = 0
#             # for metric, ylabel in zip(metrics, ylabels):
#             #     row = axs[i]
#             #     i += 1
#             #     for level, ax in zip(levels, row):
#             #         for temp in temps:
#             #             root = f'data/ablation/PointMaze/{level}/{k}/{t_label}{temp}'
#             #             plot_line_X_trials(root, line_label=f'{t_name}={temp}', ykey=metric, ax=ax, ema=0.5)
#             #         if metric == 'reachable_modes':
#             #             ax.set_ylim(0, 8 if level == 'hard' else 4)
#             #         else:
#             #             ax.set_ylim(0, 105)
#             #             ax.set_yticks(np.linspace(0, 100, 6), [*map(lambda x: '%.1f' % x, np.linspace(0, 1.0, 6))])
#             #         ax.set_title(level.capitalize())
#             #         ax.set_xlabel('Training steps')
#             #         ax.ticklabel_format(scilimits=(-4, 4), axis='x')
#             #         ax.legend(
#             #             fontsize=8, ncol=2, borderaxespad=0.2, borderpad=0.2,
#             #             handletextpad=0.3, columnspacing=0.5, handlelength=1.5
#             #         )
#             #         ax.set_ylabel(ylabel)
#             # fname = k.replace('-tanh', '') if k == 'DACER-tanh' else k
#         after_plot(
#             title=f'Quality-diversity space locations in {style}', fpath=f'figures/smbgen/{style}.png', legend_kwargs=legend_kwargs, tight_pad=1.0
#         )

# def make_smb_ablation():
#     styles = ('MarioPuzzle', 'MultiFacet')
#     algos = {
#         'DrAmt': {'name': 'DrAmort', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
#         'DrDfs': {'name': 'DrDiffus', 'dires': ('beta0.2', 'beta0.5', 'beta0.8'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
#         'DACER': {'name': 'DACER', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-|\mathcal{A}|$', r'$\hat H=-0.25|\mathcal{A}|$', r'$\hat H=0.5|\mathcal{A}|$')},
#         'SAC': {'name': 'SAC', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-|\mathcal{A}|$', r'$\hat H=-0.25|\mathcal{A}|$', r'$\hat H=0.5|\mathcal{A}|$')},
#         'SQL': {'name': 'SQL', 'dires': ('alpha0.05', 'alpha0.175', 'alpha0.3'), 'labels': (r'$\alpha=0.05$', r'$\alpha=0.175$', r'$\alpha=0.3$')},
#         'SSAC': {'name': 'S$^2$AC', 'dires': ('alpha0.1', 'alpha0.4', 'alpha0.7'), 'labels': (r'$\alpha=0.1$', r'$\alpha=0.4$', r'$\alpha=0.7$')}
#     }

#     def __collect(__folder):
#         q = []
#         d = []
#         for folder in glob.glob(os.path.join(__folder, '*')):
#             if not os.path.exists(os.path.join(folder, 'final.pt')):
#                 continue
#             data = pd.read_csv(os.path.join(folder, 'eval_log.csv'))
#             q.append(data['reward'].to_numpy()[-1])
#             d.append(data['avg-distance'].to_numpy()[-1])
#         return q, d

#     # 对每个算法分别绘制箱型图
#     for k, info in algos.items():
#         # 创建2x2的子图布局
#         fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
#         # 为每个任务和指标收集数据
#         for i, style in enumerate(styles):
#             # 收集质量数据
#             quality_data = []
#             # 收集多样性数据
#             diversity_data = []
            
#             # 为每个参数配置收集数据
#             for dire in info['dires']:
#                 q, d = __collect(f'data/ablation/smbgen/{style}/{k}/{dire}')
#                 quality_data.append(q)
#                 diversity_data.append(d)
            
#             # 绘制质量箱型图 (第1、3个子图)
#             bp_quality = axs[i*2].boxplot(quality_data, labels=info['labels'], patch_artist=True)
#             axs[i*2].set_title(f'{style} - Quality')
#             axs[i*2].set_ylabel('Quality')
            
#             # 绘制多样性箱型图 (第2、4个子图)
#             bp_diversity = axs[i*2 + 1].boxplot(diversity_data, labels=info['labels'], patch_artist=True)
#             axs[i*2 + 1].set_title(f'{style} - Diversity')
#             axs[i*2 + 1].set_ylabel('Diversity')
        
#         # 设置整体标题和布局
#         plt.suptitle(f'{info["name"]} - Quality and Diversity across Tasks', fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
        
#         # 保存图像
#         # os.makedirs('figures/smbgen/boxplots', exist_ok=True)
#         plt.savefig(f'figures/smbgen/{info['name']}_boxplot.png', dpi=300, bbox_inches='tight')
#         plt.close()

def make_smb_ablation():
    styles = ('MarioPuzzle', 'MultiFacet')
    algos = {
        'DrAmt': {'name': 'DrAmort', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
        'DrDfs': {'name': 'DrDiffus', 'dires': ('beta0.2', 'beta0.5', 'beta0.8'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
        'DACER': {'name': 'DACER', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-20$', r'$\hat H=-5$', r'$\hat H=10$')},
        'SAC': {'name': 'SAC', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-20$', r'$\hat H=-5$', r'$\hat H=10$')},
        'SQL': {'name': 'SQL', 'dires': ('alpha0.05', 'alpha0.175', 'alpha0.3'), 'labels': (r'$\alpha=0.05$', r'$\alpha=0.175$', r'$\alpha=0.3$')},
        'SSAC': {'name': 'S$^2$AC', 'dires': ('alpha0.1', 'alpha0.4', 'alpha0.7'), 'labels': (r'$\alpha=0.1$', r'$\alpha=0.4$', r'$\alpha=0.7$')}
    }

    # 定义每个算法的颜色
    colors = ['#736bd7', '#bb5b33', '#ecd37c', '#beaaff', '#ff9c72', '#338d33', '#86c7b7', '#acbbe6', '#3385a0']

    def __collect(__folder):
        q = []
        d = []
        for folder in glob.glob(os.path.join(__folder, '*')):
            if not os.path.exists(os.path.join(folder, 'final.pt')):
                continue
            data = pd.read_csv(os.path.join(folder, 'eval_log.csv'))
            q.append(data['reward'].to_numpy()[-1])
            d.append(data['avg-distance'].to_numpy()[-1])
        return q, d

    # 收集所有数据
    data_dict = {}
    for style in styles:
        data_dict[style] = {'quality': {}, 'diversity': {}}
        for k, info in algos.items():
            quality_data = []
            diversity_data = []
            for dire in info['dires']:
                q, d = __collect(f'data/ablation/smbgen/{style}/{k}/{dire}')
                quality_data.append(q)
                diversity_data.append(d)
            data_dict[style]['quality'][k] = quality_data
            data_dict[style]['diversity'][k] = diversity_data

    # 创建大图
    fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))

    # 定义子图标题
    subplot_titles = [
        'MarioPuzzle - Quality',
        'MarioPuzzle - Diversity', 
        'MultiFacet - Quality',
        'MultiFacet - Diversity'
    ]

    # 为每个子图绘制箱型图
    for i, (style, metric) in enumerate([('MarioPuzzle', 'quality'), ('MarioPuzzle', 'diversity'),
                                        ('MultiFacet', 'quality'), ('MultiFacet', 'diversity')]):
        ax = axs[i//3][i%2]
        
        # 计算每个算法箱型图的位置
        positions = []
        x_labels = []
        algo_positions = {}
        
        # 为每个算法分配位置
        algo_count = len(algos)
        for algo_idx, (k, info) in enumerate(algos.items()):
            # 每个算法占3个位置（对应3个参数）
            start_pos = algo_idx * 4  # 每个算法之间留一个空位
            algo_positions[k] = list(range(start_pos, start_pos + 3))
            positions.extend(algo_positions[k])
            x_labels.extend(info['labels'])
        
        # 绘制箱型图
        boxes = []
        for algo_idx, (k, info) in enumerate(algos.items()):
            algo_data = data_dict[style][metric][k]
            color = colors[algo_idx]
            
            # 绘制该算法的三个参数箱型图
            bp = ax.boxplot(
                algo_data, 
                positions=algo_positions[k],
                widths=0.7,
                patch_artist=True,
                labels=info['labels']
            )
            
            # 设置箱型图颜色
            for patch in bp['boxes']:
                patch.set_facecolor(color)
            
            # 记录第一个箱型图用于图例
            boxes.append(bp['boxes'][0])
        
        # 设置x轴刻度和标签
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='center')
        ax.set_title(subplot_titles[i], fontsize=14)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        if i == 2:
            ax.set_ylim((25, 50))
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.7)
        ax.grid(True, axis='x', alpha=0.7, linestyle='--')

    algo_names = [info['name'] for info in algos.values()]
    fig.legend(boxes, algo_names, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 0.05),
            ncol=6,
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True)

    plt.suptitle('Algorithm Performance Comparison Across Temperatures', fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    plt.savefig('figures/smbgen/all_ablation.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # make_mgmaze_ablation()
    # make_mgmaze_comparison()
    # make_mujoco_ablation()
    make_mujoco_comparison()
    # __make_mgmaze_dacer_itv()
    # make_smb_ablation()
