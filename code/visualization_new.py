import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from utils import macro_region

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
                     'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
output_dir = os.path.join('output', 'figures')
os.makedirs(output_dir, exist_ok=True)

print("[14/16] Spatial distribution...")
ws_detail = pd.read_csv(os.path.join('data', 'warspotting_losses.csv'))
ws_detail['date'] = pd.to_datetime(ws_detail['date'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ws_detail['region'] = ws_detail.apply(lambda r: macro_region(r['latitude'], r['longitude'], short=True), axis=1)
region_counts = ws_detail['region'].value_counts()
colors_r = ['#c0392b', '#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#95a5a6']
ax = axes[0]
bars = ax.bar(region_counts.index, region_counts.values, color=colors_r[:len(region_counts)], edgecolor='white')
ax.set_ylabel('Verified Losses')
ax.set_title('Equipment Losses by Macro Region (WarSpotting)')
for bar, val in zip(bars, region_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(val), ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
area_counts = ws_detail['nearest_location'].dropna().apply(
    lambda x: x.split(',')[-1].strip() if ',' in str(x) else str(x)).value_counts().head(10)
ax.barh(area_counts.index[::-1], area_counts.values[::-1], color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('Verified Losses')
ax.set_title('Top 10 Loss Locations')
for bar, val in zip(ax.patches, area_counts.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '14_spatial_distribution.png'))
plt.close()
print("  14_spatial_distribution.png")

print("[15/16] Observation sensitivity heatmap...")
sensitivity = pd.read_csv(os.path.join('output', 'tables', 'observation_sensitivity_overall.csv'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

pivot_ratio = sensitivity.pivot_table(values='corrected_ratio', index='p_RU', columns='p_UA', aggfunc='first')
ax = axes[0]
im = ax.imshow(pivot_ratio.values, cmap='RdYlGn_r', aspect='auto', vmin=0.5, vmax=2.5)
ax.set_xticks(range(len(pivot_ratio.columns)))
ax.set_xticklabels([f'{c:.1f}' for c in pivot_ratio.columns])
ax.set_yticks(range(len(pivot_ratio.index)))
ax.set_yticklabels([f'{r:.1f}' for r in pivot_ratio.index])
ax.set_xlabel('p_observed_Ukraine')
ax.set_ylabel('p_observed_Russia')
ax.set_title('Corrected RU/UA Ratio (observed = 2.06)')
for i, p_ru in enumerate(pivot_ratio.index):
    for j, p_ua in enumerate(pivot_ratio.columns):
        val = pivot_ratio.values[i, j]
        color = 'white' if val < 1.0 or val > 1.8 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[1]
p_ru_vals = pivot_ratio.index.values
for p_ua in pivot_ratio.columns:
    ratios = [pivot_ratio.loc[p_ru, p_ua] for p_ru in p_ru_vals]
    ax.plot(p_ru_vals, ratios, 'o-', linewidth=1, markersize=4, alpha=0.7, label=f'p_UA={p_ua:.1f}')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Ratio = 1')
ax.fill_between(p_ru_vals, 0, 1, alpha=0.1, color='blue', label='UA > RU region')
ax.set_xlabel('p_observed_Russia')
ax.set_ylabel('Corrected RU/UA Ratio')
ax.set_title('Ratio Reversal Boundary')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '15_sensitivity_heatmap.png'))
plt.close()
print("  15_sensitivity_heatmap.png")

print("[16/16] Weekly Bootstrap distribution...")
boot_dist = pd.read_csv(os.path.join('output', 'tables', 'weekly_bootstrap_distribution.csv'))
boot_summary = pd.read_csv(os.path.join('output', 'tables', 'weekly_bootstrap_bca.csv'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
rates = boot_dist['rate_per_day'].values
ax.hist(rates, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(x=boot_summary['theta_hat_rate_per_day'].values[0], color='black', linewidth=2,
           label=f"MLE = {boot_summary['theta_hat_rate_per_day'].values[0]:.3f}")
ax.axvline(x=boot_summary['bca_ci_low_per_day'].values[0], color='red', linestyle='--', linewidth=1.5,
           label=f"BCa CI [{boot_summary['bca_ci_low_per_day'].values[0]:.3f}, {boot_summary['bca_ci_high_per_day'].values[0]:.3f}]")
ax.axvline(x=boot_summary['bca_ci_high_per_day'].values[0], color='red', linestyle='--', linewidth=1.5)
ax.axvline(x=boot_summary['percentile_ci_low_per_day'].values[0], color='green', linestyle=':', linewidth=1,
           label=f"Pct CI [{boot_summary['percentile_ci_low_per_day'].values[0]:.3f}, {boot_summary['percentile_ci_high_per_day'].values[0]:.3f}]")
ax.axvline(x=boot_summary['percentile_ci_high_per_day'].values[0], color='green', linestyle=':', linewidth=1)
ax.set_xlabel('Rate per Day')
ax.set_ylabel('Density')
ax.set_title(f"Weekly Block Bootstrap (B={int(boot_summary['bootstrap_reps'].values[0])}, "
             f"n_weeks={int(boot_summary['n_weeks'].values[0])})")
ax.legend(fontsize=8)

ax = axes[1]
type_boot = pd.read_csv(os.path.join('output', 'tables', 'weekly_bootstrap_by_type.csv'))
ax.barh(type_boot['equipment_type'], type_boot['bootstrap_se'], color='steelblue', alpha=0.7, edgecolor='white')
ax.set_xlabel('Bootstrap Standard Error')
ax.set_title('Weekly Block Bootstrap SE by Equipment Type')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '16_weekly_bootstrap.png'))
plt.close()
print("  16_weekly_bootstrap.png")

print("\nCharts 14-16 generated!")
