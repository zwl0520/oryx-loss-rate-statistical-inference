"""
双方对比可视化
=============
在原有 9 张图的基础上，新增俄乌双方的直接统计对比图。
"""

import pandas as pd, numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
output_dir = os.path.join('output', 'figures')
os.makedirs(output_dir, exist_ok=True)

# 加载数据
daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'), parse_dates=['Date'])
monthly = pd.read_csv(os.path.join('data', 'monthly_oryx_both_sides.csv'))
monthly['month'] = monthly['month'].astype(str)

# ===========================
# 图10: 双方日均损失时序对比
# ===========================
print("[10/16] 双方损失时序对比...")
fig, axes = plt.subplots(3, 1, figsize=(14, 13))

# 10.1 双方日损失 30天MA
ax = axes[0]
for side, color, label in [('Russia_Total', '#c0392b', 'Russia (RU)'), ('Ukraine_Total', '#2980b9', 'Ukraine (UA)')]:
    ax.plot(daily['Date'], daily[side].rolling(30).mean(), color=color, linewidth=1.5, label=f'{label} 30-day MA')
    ax.axhline(y=daily[side].mean(), color=color, linestyle='--', alpha=0.4, label=f'{label} Mean ({daily[side].mean():.1f}/day)')
ax.set_ylabel('Daily Losses (30-day MA)')
ax.set_title('Image-Verified Equipment Losses: Russia vs Ukraine (WarSpotting/Oryx)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# 10.2 双方累计损失
ax = axes[1]
ax.fill_between(daily['Date'], daily['Russia_Total'].cumsum(), alpha=0.3, color='#c0392b')
ax.fill_between(daily['Date'], daily['Ukraine_Total'].cumsum(), alpha=0.3, color='#2980b9')
ax.plot(daily['Date'], daily['Russia_Total'].cumsum(), color='#c0392b', linewidth=1.5, label=f'Russia ({daily['Russia_Total'].sum():,})')
ax.plot(daily['Date'], daily['Ukraine_Total'].cumsum(), color='#2980b9', linewidth=1.5, label=f'Ukraine ({daily["Ukraine_Total"].sum():,})')
ax.set_ylabel('Cumulative Losses')
ax.set_title('Cumulative Verified Equipment Losses')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# 10.3 月度比率变化
ax = axes[2]
ratio = monthly['RU_Total'] / monthly['UA_Total'].replace(0, np.nan)
months = monthly['month'].values
colors = ['#c0392b' if r >= 1 else '#2980b9' for r in ratio]
ax.bar(range(len(months)), ratio.values, color=colors, alpha=0.7, edgecolor='white')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Ratio = 1 (equal losses)')
ax.axhline(y=ratio.mean(), color='gray', linestyle=':', linewidth=1, label=f'Mean ratio = {ratio.mean():.2f}')
ax.set_xticks(range(0, len(months), 6))
ax.set_xticklabels(months[::6], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Loss Ratio RU/UA')
ax.set_title('Monthly Equipment Loss Ratio: Russia / Ukraine (>1 = Russia loses more)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '10_both_sides_timeseries.png'))
plt.close()
print("  ✓ 10_both_sides_timeseries.png")

# ===========================
# 图11: 双方 CI 对比森林图
# ===========================
print("[11/16] 双方 CI 森林图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

categories = ['Total', 'Destroyed', 'Damaged', 'Abandoned', 'Captured',
              'Tanks', 'AFV', 'IFV', 'APC', 'Artillery', 'Aircraft', 'Vehicles']

def get_ci(data, alpha=0.05):
    n = len(data)
    lam = np.mean(data)
    Y = np.sum(data)
    se = np.sqrt(lam / n)
    low = max(0, lam - 1.96 * se)
    upp = lam + 1.96 * se
    e_low = stats.chi2.ppf(alpha/2, max(2*int(Y), 1)) / (2*n) if Y > 0 else 0
    e_upp = stats.chi2.ppf(1 - alpha/2, 2*(int(Y)+1)) / (2*n)
    return lam, se, low, upp, e_low, e_upp

# 俄方 CI
ax = axes[0]
for i, (label, col) in enumerate([
    ('Total', 'Russia_Total'), ('Destroyed', 'Russia_Destroyed'), ('Damaged', 'Russia_Damaged'),
    ('Abandoned', 'Russia_Abandoned'), ('Captured', 'Russia_Captured'),
    ('Tanks', 'Russia_Tanks'), ('AFV', 'Russia_AFV'), ('IFV', 'Russia_IFV'),
    ('APC', 'Russia_APC'), ('Artillery', 'Russia_Artillery'), ('Aircraft', 'Russia_Aircraft'),
    ('Vehicles', 'Russia_Vehicles'),
]):
    lam, se, wl, wu, el, eu = get_ci(daily[col].values)
    ax.plot([el, eu], [i, i], 'r-', linewidth=4, alpha=0.7, label='Exact CI' if i == 0 else '')
    ax.plot([wl, wu], [i, i], 'b-', linewidth=2, alpha=0.5, label='Wald CI' if i == 0 else '')
    ax.plot(lam, i, 'ko', markersize=6)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)
ax.set_xlabel('Daily Loss Rate λ')
ax.set_title('Russia: 95% CI by Category')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# 乌方 CI
ax = axes[1]
for i, (label, col) in enumerate([
    ('Total', 'Ukraine_Total'), ('Destroyed', 'Ukraine_Destroyed'), ('Damaged', 'Ukraine_Damaged'),
    ('Abandoned', 'Ukraine_Abandoned'), ('Captured', 'Ukraine_Captured'),
    ('Tanks', 'Ukraine_Tanks'), ('AFV', 'Ukraine_AFV'), ('IFV', 'Ukraine_IFV'),
    ('APC', 'Ukraine_APC'), ('Artillery', 'Ukraine_Artillery'), ('Aircraft', 'Ukraine_Aircraft'),
    ('Vehicles', 'Ukraine_Vehicles'),
]):
    lam, se, wl, wu, el, eu = get_ci(daily[col].values)
    ax.plot([el, eu], [i, i], 'r-', linewidth=4, alpha=0.7, label='Exact CI' if i == 0 else '')
    ax.plot([wl, wu], [i, i], 'b-', linewidth=2, alpha=0.5, label='Wald CI' if i == 0 else '')
    ax.plot(lam, i, 'ko', markersize=6)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)
ax.set_xlabel('Daily Loss Rate λ')
ax.set_title('Ukraine: 95% CI by Category')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '11_both_sides_ci_forest.png'))
plt.close()
print("  ✓ 11_both_sides_ci_forest.png")

# ===========================
# 图12: 双方装备类型构成对比
# ===========================
print("[12/16] 双方装备类型构成对比...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

eq_names = ['Tanks', 'IFV', 'AFV', 'APC', 'Artillery', 'Vehicles', 'Aircraft', 'Engineering', 'Antiair', 'Logistics']
ru_eq_vals = [daily[f'Russia_{e}'].sum() for e in eq_names]
ua_eq_vals = [daily[f'Ukraine_{e}'].sum() for e in eq_names]

# 绝对数量
ax = axes[0]
x = np.arange(len(eq_names))
w = 0.35
bars1 = ax.bar(x - w/2, ru_eq_vals, w, color='#c0392b', alpha=0.8, label='Russia', edgecolor='white')
bars2 = ax.bar(x + w/2, ua_eq_vals, w, color='#2980b9', alpha=0.8, label='Ukraine', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(eq_names, rotation=45, ha='right')
ax.set_ylabel('Total Verified Losses')
ax.set_title('Equipment Losses by Type: Russia vs Ukraine (Absolute)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 比率
ax = axes[1]
ratios = [ru / max(ua, 1) for ru, ua in zip(ru_eq_vals, ua_eq_vals)]
colors_bar = ['#c0392b' if r >= 1 else '#2980b9' for r in ratios]
bars = ax.bar(eq_names, ratios, color=colors_bar, alpha=0.8, edgecolor='white')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Loss Ratio RU/UA')
ax.set_title('Equipment Loss Ratio by Type (>1 = Russia loses more)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
# 标注数值
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{ratio:.1f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '12_equipment_type_comparison.png'))
plt.close()
print("  ✓ 12_equipment_type_comparison.png")

# ===========================
# 图13: 双方损失状态构成饼图
# ===========================
print("[13/16] 双方损失状态对比...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

status_labels = ['Destroyed', 'Damaged', 'Abandoned', 'Captured']
status_colors = ['#e74c3c', '#f39c12', '#95a5a6', '#e67e22']

for ax, prefix, title in [(axes[0], 'Russia_', 'Russia'), (axes[1], 'Ukraine_', 'Ukraine')]:
    vals = [daily[f'{prefix}{s}'].sum() for s in status_labels]
    wedges, texts, autotexts = ax.pie(vals, labels=status_labels, autopct='%1.1f%%',
                                       colors=status_colors, startangle=90,
                                       explode=(0.02, 0.02, 0.02, 0.02))
    ax.set_title(f'{title}: Loss by Status\n(Total: {sum(vals):,})')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '13_both_sides_status_pie.png'))
plt.close()
print("  ✓ 13_both_sides_status_pie.png")

print(f"\n双方对比可视化完成！新增4张图表 (10-13)")
