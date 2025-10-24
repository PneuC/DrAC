import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
# import numpy as np

styles = ('MarioPuzzle', 'MultiFacet')
algos = {
    'DrAmt': {'name': 'DrAmort', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
    'DrDfs': {'name': 'DrDiffus', 'dires': ('beta0.2', 'beta0.5', 'beta0.8'), 'labels': (r'$\beta=0.2$', r'$\beta=0.5$', r'$\beta=0.8$')},
    'DACER': {'name': 'DACER', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-|\mathcal{A}|$', r'$\hat H=-0.25|\mathcal{A}|$', r'$\hat H=0.5|\mathcal{A}|$')},
    'SAC': {'name': 'SAC', 'dires': ('low', 'mid', 'high'), 'labels': (r'$\hat H=-|\mathcal{A}|$', r'$\hat H=-0.25|\mathcal{A}|$', r'$\hat H=0.5|\mathcal{A}|$')},
    'SQL': {'name': 'SQL', 'dires': ('alpha0.05', 'alpha0.175', 'alpha0.3'), 'labels': (r'$\alpha=0.05$', r'$\alpha=0.175$', r'$\alpha=0.3$')},
    'SSAC': {'name': 'S$^2$AC', 'dires': ('alpha0.1', 'alpha0.4', 'alpha0.7'), 'labels': (r'$\alpha=0.1$', r'$\alpha=0.4$', r'$\alpha=0.7$')}
}

# 定义每个算法的颜色
colors = ['#736bd7', '#bb5b33', '#ecd37c', '#beaaff', '#ff9c72', '#338d33', '#86c7b7', '#acbbe6', '#3385a0']

def collect(folder):
    """收集文件夹中的质量和多样性数据"""
    q = []
    d = []
    for subfolder in glob.glob(os.path.join(folder, '*')):
        if not os.path.exists(os.path.join(subfolder, 'final.pt')):
            continue
        try:
            data = pd.read_csv(os.path.join(subfolder, 'eval_log.csv'))
            q.append(data['reward'].to_numpy()[-1])
            d.append(data['avg-distance'].to_numpy()[-1])
        except Exception as e:
            print(f"Error reading {subfolder}: {e}")
    return q, d

def calculate_hv(points, ref_point):
    """计算超体积指标(HV)"""
    if len(points) < 2:
        return 0.0
    
    # 确保所有点都在参考点之上（对于最小化问题，我们需要转换）
    # 这里假设我们是在最大化质量和多样性
    filtered_points = [p for p in points if p[0] > ref_point[0] and p[1] > ref_point[1]]
    
    if len(filtered_points) < 2:
        return 0.0
    
    # 找到非支配点（帕累托前沿）
    non_dominated = []
    for i, p1 in enumerate(filtered_points):
        dominated = False
        for j, p2 in enumerate(filtered_points):
            if i != j and p2[0] >= p1[0] and p2[1] >= p1[1] and (p2[0] > p1[0] or p2[1] > p1[1]):
                dominated = True
                break
        if not dominated:
            non_dominated.append(p1)
    
    if len(non_dominated) < 2:
        # 如果只有一个非支配点，计算矩形面积
        if len(non_dominated) == 1:
            p = non_dominated[0]
            return (p[0] - ref_point[0]) * (p[1] - ref_point[1])
        return 0.0
    
    # 按质量排序
    non_dominated.sort(key=lambda x: x[0])
    
    # 计算HV（使用梯形法则）
    hv = 0.0
    for i in range(len(non_dominated)):
        if i == 0:
            # 第一个点
            width = non_dominated[i][0] - ref_point[0]
            height = non_dominated[i][1] - ref_point[1]
        else:
            # 后续点
            width = non_dominated[i][0] - non_dominated[i-1][0]
            height = non_dominated[i][1] - ref_point[1]
        hv += width * height
    
    return hv

# 收集所有数据并确定全局最差参考点
all_points_by_style = {'MarioPuzzle': [], 'MultiFacet': []}
all_quality_by_style = {'MarioPuzzle': [], 'MultiFacet': []}
all_diversity_by_style = {'MarioPuzzle': [], 'MultiFacet': []}

print("Collecting all data to determine reference points...")
for style in styles:
    for k, info in algos.items():
        for dire in info['dires']:
            q, d = collect(f'data/ablation/smbgen/{style}/{k}/{dire}')
            for i in range(len(q)):
                all_points_by_style[style].append((q[i], d[i]))
                all_quality_by_style[style].append(q[i])
                all_diversity_by_style[style].append(d[i])
    
    print(f"{style}: Collected {len(all_points_by_style[style])} data points")

# 确定每个任务的参考点（最差值）
ref_points = {}
for style in styles:
    if all_quality_by_style[style] and all_diversity_by_style[style]:
        min_quality = min(all_quality_by_style[style])
        min_diversity = min(all_diversity_by_style[style])
        ref_points[style] = (min_quality, min_diversity)
        print(f"{style} reference point: Quality={min_quality:.4f}, Diversity={min_diversity:.4f}")
    else:
        ref_points[style] = (0, 0)
        print(f"{style}: No data found, using default reference point (0,0)")

# 计算每个算法的HV值
hv_results = {}

for style in styles:
    hv_results[style] = {}
    ref_point = ref_points[style]
    
    for k, info in algos.items():
        # 收集该算法所有参数配置的数据
        all_points = []
        
        for dire in info['dires']:
            q, d = collect(f'data/ablation/smbgen/{style}/{k}/{dire}')
            # 将质量和多样性组合成点
            for i in range(len(q)):
                all_points.append((q[i], d[i]))
        
        # 计算HV
        hv_value = calculate_hv(all_points, ref_point=ref_point)
        hv_results[style][k] = hv_value
        
        print(f"{style} - {info['name']}: HV = {hv_value:.4f} (ref: {ref_point[0]:.2f}, {ref_point[1]:.2f})")

# 可视化HV结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4))

# MarioPuzzle的HV结果
algo_names = [info['name'] for info in algos.values()]
mario_hv = [hv_results['MarioPuzzle'][k] for k in algos.keys()]
bars1 = ax1.bar(algo_names, mario_hv, color=colors)
ax1.set_title(f'MarioPuzzle')
ax1.set_ylabel('HV Value')
# ax1.set_xlabel('Algorithms')
ax1.set_ylim((0, 60000))
ax1.grid(False, axis='x')
ax1.grid(True, axis='y')
ax1.tick_params(axis='x', rotation=45)
ax1.set_axisbelow(True)

# 在柱状图上添加数值标签
for bar, hv in zip(bars1, mario_hv):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(mario_hv),
             f'{hv:.1f}', ha='center', va='bottom')

# MultiFacet的HV结果
multifacet_hv = [hv_results['MultiFacet'][k] for k in algos.keys()]
bars2 = ax2.bar(algo_names, multifacet_hv, color=colors)
ax2.set_title(f'MultiFacet')
ax2.set_ylabel('HV Value')
# ax2.set_xlabel('Algorithms')
ax2.set_ylim((0, 12000))
ax2.grid(True, axis='y')
ax2.grid(False, axis='x')
ax2.tick_params(axis='x', rotation=45)
ax2.set_axisbelow(True)

# 在柱状图上添加数值标签
for bar, hv in zip(bars2, multifacet_hv):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(multifacet_hv),
             f'{hv:.1f}', ha='center', va='bottom')

plt.suptitle('Hypervolume (HV) Comparison Across Algorithms and Tasks', fontsize=14)
plt.tight_layout()

# 保存图像
plt.savefig('figures/smbgen/hv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建汇总表格
print("\nHV Results Summary (using worst observed values as reference):")
print("=" * 80)
print(f"{'Algorithm':<15} {'MarioPuzzle':<20} {'MultiFacet':<20}")
print(f"{'':<15} {'HV Value':<10} {'% of Best':<10} {'HV Value':<10} {'% of Best':<10}")
print("-" * 80)

# 计算每个任务的最佳HV值
best_mario_hv = max(hv_results['MarioPuzzle'].values())
best_multifacet_hv = max(hv_results['MultiFacet'].values())

for k, info in algos.items():
    mario_hv = hv_results['MarioPuzzle'][k]
    multifacet_hv = hv_results['MultiFacet'][k]
    mario_percent = (mario_hv / best_mario_hv * 100) if best_mario_hv > 0 else 0
    multifacet_percent = (multifacet_hv / best_multifacet_hv * 100) if best_multifacet_hv > 0 else 0
    
    print(f"{info['name']:<15} {mario_hv:<10.4f} {mario_percent:<10.1f}% {multifacet_hv:<10.4f} {multifacet_percent:<10.1f}%")

print("=" * 80)

# 找出每个任务中表现最好的算法
best_mario = max(hv_results['MarioPuzzle'], key=hv_results['MarioPuzzle'].get)
best_multifacet = max(hv_results['MultiFacet'], key=hv_results['MultiFacet'].get)

print(f"\nBest performing algorithm:")
print(f"  MarioPuzzle: {algos[best_mario]['name']} (HV = {hv_results['MarioPuzzle'][best_mario]:.4f})")
print(f"  MultiFacet: {algos[best_multifacet]['name']} (HV = {hv_results['MultiFacet'][best_multifacet]:.4f})")