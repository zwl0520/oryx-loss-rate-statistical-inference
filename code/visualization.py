import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

from utils import VIZ_PHASES

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'figure.figsize': (12, 6),
})

output_dir = os.path.join('output', 'figures')
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Generating visualizations...")

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_by_status = pd.read_csv(os.path.join('data', 'daily_by_status_ws.csv'), index_col=0, parse_dates=True)
daily_ua = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'), parse_dates=['date'])
ci_df = pd.read_csv(os.path.join('output', 'tables', 'ci_comparison.csv'))
boot_dist = pd.read_csv(os.path.join('output', 'tables', 'bootstrap_distributions.csv'))

print("  [1/9] Daily loss time series...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
ax.bar(daily_ws['date'], daily_ws['total_losses'], alpha=0.3, color='steelblue', width=1, label='Daily Verified Losses')
ax.plot(daily_ws['date'], daily_ws['total_losses'].rolling(30).mean(), color='darkred', linewidth=1.5, label='30-day MA')
ax.axhline(y=daily_ws['total_losses'].mean(), color='black', linestyle='--', alpha=0.5,
           label=f'Overall Mean ({daily_ws["total_losses"].mean():.1f}/day)')
ax.set_ylabel('Equipment Losses')
ax.set_title('Daily Image-Verified Russian Equipment Losses (WarSpotting/Oryx)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ws_tank_daily = daily_by_type['Tanks']
min_days = min(len(ws_tank_daily), len(daily_ua['Tanks']))
dates_aligned = daily_ws['date'].values[:min_days]
ax.plot(dates_aligned, ws_tank_daily.values[:min_days].cumsum(), 'steelblue', linewidth=1.5, label='WarSpotting Verified (Tanks)')
ax.plot(dates_aligned, np.cumsum(daily_ua['Tanks'].values[:min_days]), 'darkorange', linewidth=1.5, label='Ukraine Official Claims (Tanks)')
ax.set_ylabel('Cumulative Tank Losses')
ax.set_title('Cumulative Russian Tank Losses: Verified vs Claimed')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
cumsum_by_type = daily_by_type.cumsum()
colors = plt.cm.tab10(np.linspace(0, 1, 6))
for i, col in enumerate(['Tanks', 'Infantry fighting vehicles', 'Self-propelled artillery',
                          'Towed artillery', 'Drones', 'Transport']):
    if col in cumsum_by_type.columns:
        ax.fill_between(daily_by_type.index, cumsum_by_type[col].values, alpha=0.6, color=colors[i], label=col)
ax.set_ylabel('Cumulative Losses')
ax.set_xlabel('Date')
ax.set_title('Cumulative Verified Russian Equipment Losses by Type')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '01_daily_loss_timeseries.png'))
plt.close()
print("    01_daily_loss_timeseries.png")

print("  [2/9] Phase boxplot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

phase_data = []
phase_labels = []
for label, (start, end) in VIZ_PHASES.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    data = daily_ws[mask]['total_losses'].values
    if len(data) > 0:
        phase_data.append(data)
        phase_labels.append(label)

bp = axes[0].boxplot(phase_data, labels=phase_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(phase_data)))):
    patch.set_facecolor(color)
axes[0].set_ylabel('Daily Losses')
axes[0].set_title('Daily Equipment Loss Distribution by War Phase')
axes[0].grid(True, alpha=0.3)

means = [np.mean(d) for d in phase_data]
ses = [np.std(d, ddof=1) / np.sqrt(len(d)) for d in phase_data]
axes[1].bar(phase_labels, means, yerr=1.96 * np.array(ses), capsize=5,
            color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(phase_data))), edgecolor='black', alpha=0.8)
axes[1].set_ylabel('Mean Daily Losses (with 95% CI)')
axes[1].set_title('Mean Daily Equipment Loss Rate by War Phase')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '02_phase_boxplot.png'))
plt.close()
print("    02_phase_boxplot.png")

print("  [3/9] CI forest plot...")
fig, ax = plt.subplots(figsize=(12, 7))
n_items = len(ci_df)
y_pos = range(n_items)

for i, (_, row) in enumerate(ci_df.iterrows()):
    ax.plot([row['Wald_Lower'], row['Wald_Upper']], [i + 0.2, i + 0.2], 'b-', linewidth=3, alpha=0.7, label='Wald' if i == 0 else '')
    ax.plot([row['Score_Lower'], row['Score_Upper']], [i, i], 'g-', linewidth=3, alpha=0.7, label='Score' if i == 0 else '')
    ax.plot([row['Exact_Lower'], row['Exact_Upper']], [i - 0.2, i - 0.2], 'r-', linewidth=3, alpha=0.7, label='Exact' if i == 0 else '')
    ax.plot(row['MLE'], i, 'ko', markersize=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(ci_df['Category'])
ax.set_xlabel('Daily Loss Rate (items/day)')
ax.set_title('95% Confidence Intervals: Wald vs Score vs Exact')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '03_ci_forest_plot.png'))
plt.close()
print("    03_ci_forest_plot.png")

print("  [4/9] Bootstrap distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(boot_dist['IID_Bootstrap_Means'], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(x=np.percentile(boot_dist['IID_Bootstrap_Means'], 2.5), color='red', linestyle='--')
ax.axvline(x=np.percentile(boot_dist['IID_Bootstrap_Means'], 97.5), color='red', linestyle='--', label='2.5%/97.5%')
ax.axvline(x=np.mean(boot_dist['IID_Bootstrap_Means']), color='black', linewidth=1.5, label=f'Mean={np.mean(boot_dist["IID_Bootstrap_Means"]):.3f}')
ax.set_xlabel('Bootstrap Mean')
ax.set_ylabel('Density')
ax.set_title('IID Bootstrap Distribution (B=2000)')
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(boot_dist['Block_Bootstrap_Means'], bins=50, density=True, alpha=0.7, color='darkorange', edgecolor='white')
ax.axvline(x=np.percentile(boot_dist['Block_Bootstrap_Means'], 2.5), color='red', linestyle='--')
ax.axvline(x=np.percentile(boot_dist['Block_Bootstrap_Means'], 97.5), color='red', linestyle='--', label='2.5%/97.5%')
ax.axvline(x=np.mean(boot_dist['Block_Bootstrap_Means']), color='black', linewidth=1.5, label=f'Mean={np.mean(boot_dist["Block_Bootstrap_Means"]):.3f}')
ax.set_xlabel('Bootstrap Mean')
ax.set_ylabel('Density')
ax.set_title('Block Bootstrap Distribution (b=11, B=2000)')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '04_bootstrap_distribution.png'))
plt.close()
print("    04_bootstrap_distribution.png")

print("  [5/9] Equipment composition...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top_types = ['Tanks', 'Infantry fighting vehicles', 'Transport', 'Self-propelled artillery',
             'Towed artillery', 'Drones', 'Engineering', 'Anti-aircraft systems']
available_types = [t for t in top_types if t in daily_by_type.columns]

ax = axes[0]
y_data = daily_by_type[available_types].rolling(30).mean()
ax.stackplot(daily_by_type.index, *[y_data[col].values for col in available_types],
             labels=available_types, alpha=0.8, colors=plt.cm.tab10(np.linspace(0, 1, len(available_types))))
ax.set_ylabel('Daily Losses (30-day MA)')
ax.set_title('Equipment Loss Composition Over Time (30-day MA)')
ax.legend(loc='upper left', fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1]
props = y_data.div(y_data.sum(axis=1), axis=0)
ax.stackplot(daily_by_type.index, *[props[col].values for col in available_types],
             labels=available_types, alpha=0.8, colors=plt.cm.tab10(np.linspace(0, 1, len(available_types))))
ax.set_ylabel('Proportion')
ax.set_title('Equipment Loss Composition (Proportion, 30-day MA)')
ax.legend(loc='upper left', fontsize=7, ncol=2)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '05_equipment_composition.png'))
plt.close()
print("    05_equipment_composition.png")

print("  [6/9] Poisson QQ plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (title, phase_key) in zip(axes.flatten(), [
    ('Phase 1 (Initial Invasion)', ('2022-02-24', '2022-04-30')),
    ('Phase 3 (Counteroffensive)', ('2022-09-01', '2022-12-31')),
    ('Phase 7 (Spring 2024)', ('2024-03-01', '2024-07-31')),
    ('Phase 9 (Recent)', ('2025-01-01', '2026-06-30')),
]):
    start, end = phase_key
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    data = daily_ws[mask]['total_losses'].values
    lambda_hat = np.mean(data)
    n = len(data)
    empirical = np.sort(data)
    prob_levels = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.poisson.ppf(prob_levels, lambda_hat)
    ax.scatter(theoretical, empirical, alpha=0.5, s=15, color='steelblue')
    max_val = max(theoretical.max(), empirical.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
    ax.set_xlabel('Theoretical Poisson Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title(f'{title} (lambda_hat={lambda_hat:.1f})')
    ax.grid(True, alpha=0.3)

plt.suptitle('Poisson Q-Q Plots by War Phase', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, '06_poisson_qqplot.png'))
plt.close()
print("    06_poisson_qqplot.png")

print("  [7/9] CI width comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

categories = ci_df['Category'].values
wald_width = ci_df['Wald_Upper'].values - ci_df['Wald_Lower'].values
score_width = ci_df['Score_Upper'].values - ci_df['Score_Lower'].values
exact_width = ci_df['Exact_Upper'].values - ci_df['Exact_Lower'].values

x = np.arange(len(categories))
width = 0.25
ax.bar(x - width, wald_width, width, label='Wald', color='steelblue', alpha=0.8)
ax.bar(x, score_width, width, label='Score', color='seagreen', alpha=0.8)
ax.bar(x + width, exact_width, width, label='Exact', color='darkorange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('CI Width')
ax.set_title('Confidence Interval Width Comparison')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '07_ci_width_comparison.png'))
plt.close()
print("    07_ci_width_comparison.png")

print("  [8/9] Claims vs verified...")
fig, ax = plt.subplots(figsize=(10, 6))

ws_type_monthly = daily_by_type.resample('M').sum()
ua_monthly = daily_ua.set_index('date').resample('M').sum()

if 'Tanks' in ws_type_monthly.columns and 'Tanks' in ua_monthly.columns:
    common_months = ws_type_monthly.index.intersection(ua_monthly.index)
    ws_tank_monthly = ws_type_monthly.loc[common_months, 'Tanks']
    ua_tank_monthly = ua_monthly.loc[common_months, 'Tanks']
    ax.bar(common_months, ws_tank_monthly.values, alpha=0.7, color='steelblue', label='WarSpotting Verified (Tanks)', width=20)
    ax.bar(common_months, ua_tank_monthly.values, alpha=0.5, color='darkorange', label='Ukraine Official Claims (Tanks)', width=20)
    ax.set_ylabel('Monthly Tank Losses')
    ax.set_title('Monthly Russian Tank Losses: Verified vs Claimed')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '08_claims_vs_verified.png'))
plt.close()
print("    08_claims_vs_verified.png")

print("  [9/9] Status and type...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

status_counts = daily_by_status.sum()
colors_pie = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6']
wedges, texts, autotexts = axes[0].pie(
    status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
    colors=colors_pie[:len(status_counts)], startangle=90, explode=(0.02, 0.02, 0.02, 0.02))
axes[0].set_title('Equipment Loss by Status (WarSpotting)')

top10_types = daily_by_type.sum().sort_values(ascending=True).tail(10)
bars = axes[1].barh(top10_types.index, top10_types.values, color=plt.cm.viridis(np.linspace(0.2, 0.9, 10)))
axes[1].set_xlabel('Total Count')
axes[1].set_title('Top 10 Equipment Types by Verified Losses')
for bar, val in zip(bars, top10_types.values):
    axes[1].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, str(int(val)), va='center', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '09_status_and_type.png'))
plt.close()
print("    09_status_and_type.png")

print("\n" + "=" * 60)
print("All 9 charts generated in output/figures/")
