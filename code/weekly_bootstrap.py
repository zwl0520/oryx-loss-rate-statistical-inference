"""
周度 Block Bootstrap 模块
========================
在参考仓库基础上改进的周度 Block Bootstrap 分析。

改进点：
1. 使用周度聚合（每7天一块更自然，块长度选择更稳健）
2. B=10000 次 Bootstrap（vs 本地项目的 2000 次）
3. 完整的 Jackknife + BCa 区间
4. 同时输出日率、月率（×30天的可解释性更好）
5. 按装备类型的分类型 Bootstrap
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, sys, math, random

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载数据并生成周度序列
# ===========================
print("=" * 70)
print("周度 Block Bootstrap 分析 (B=10000)")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_ws['week'] = daily_ws['date'].dt.isocalendar().year.astype(str) + '-W' + \
                   daily_ws['date'].dt.isocalendar().week.astype(str).str.zfill(2)

weekly = daily_ws.groupby('week').agg(
    week_start=('date', 'min'),
    loss_count=('total_losses', 'sum'),
).reset_index()
weekly = weekly.sort_values('week_start').reset_index(drop=True)

n_weeks = len(weekly)
counts = weekly['loss_count'].values
print(f"周数: {n_weeks}")
print(f"周均损失: {np.mean(counts):.1f} 件/周")
print(f"日均损失 (周度基准): {np.sum(counts) / (7 * n_weeks):.3f} 件/天")

# ===========================
# 2. Block Bootstrap (按周抽取)
# ===========================
B = 10000
SEED = 20260605
ALPHA = 0.05

random.seed(SEED)
np.random.seed(SEED)

# 统计量：日均损失率 λ̂
theta_hat = np.sum(counts) / (7 * n_weeks)
print(f"\nθ̂ (MLE, 周度基准) = {theta_hat:.4f} 件/天")

# ===========================
# 周度 Block Bootstrap（Moving Block Bootstrap）
# ===========================
# 块长度选择：基于周度 ACF 衰减
def choose_block_length_weekly(data, max_lag=12):
    """周度数据的块长度选择（基于 ACF 衰减到 2/√n 以下）"""
    n = len(data)
    x = data - np.mean(data)
    var = np.sum(x**2)
    if var < 1e-10:
        return 2
    threshold = 2.0 / np.sqrt(n)
    for lag in range(1, min(max_lag, n // 4)):
        acf_val = np.sum(x[lag:] * x[:-lag]) / var
        if abs(acf_val) < threshold:
            return max(lag, 2)
    return max(int(n ** (1/3)), 2)

b_weekly = choose_block_length_weekly(counts)
print(f"周度数据长度: n = {n_weeks} 周")
print(f"周度块长度: b = {b_weekly} 周")

# 显示周度 ACF
x_w = counts - np.mean(counts)
var_w = np.sum(x_w**2)
acf_w = []
for lag in range(1, min(9, n_weeks//4)):
    acf_w.append(round(np.sum(x_w[lag:] * x_w[:-lag]) / var_w, 3))
print(f"周度自相关 (lag 1-{len(acf_w)}): {acf_w}")

# Moving Block Bootstrap on weekly data
boot_rates = np.zeros(B)
k_blocks = int(np.ceil(n_weeks / b_weekly))
for b in range(B):
    # 有放回地抽取 k 个连续块
    bs_weekly = np.zeros(n_weeks)
    pos = 0
    for _ in range(k_blocks):
        start = np.random.randint(0, n_weeks - b_weekly + 1)
        block = counts[start:start + b_weekly]
        end = min(pos + len(block), n_weeks)
        bs_weekly[pos:end] = block[:end - pos]
        pos = end
        if pos >= n_weeks:
            break
    boot_rates[b] = np.sum(bs_weekly) / (7 * n_weeks)

boot_mean = np.mean(boot_rates)
boot_se = np.std(boot_rates, ddof=1)
boot_percentile_ci = (np.percentile(boot_rates, 2.5), np.percentile(boot_rates, 97.5))

print(f"Bootstrap Mean: {boot_mean:.4f}")
print(f"Bootstrap SE (周度块): {boot_se:.4f}")
print(f"Percentile 95% CI: [{boot_percentile_ci[0]:.4f}, {boot_percentile_ci[1]:.4f}]")

# 对比本地项目日度结果
# 读取日度 Bootstrap 结果进行对比
try:
    bs_results = pd.read_csv(os.path.join('output', 'tables', 'bootstrap_results.csv'))
    daily_se = bs_results['Bootstrap_SE'].iloc[0] if 'Bootstrap_SE' in bs_results.columns else None
except:
    daily_se = None

print(f"\n对比:")
if daily_se is not None:
    print(f"  日度 MBB SE: {daily_se:.4f} (WarSpotting, B=2000)")
    print(f"  周度 MBB SE: {boot_se:.4f} (WarSpotting, b={b_weekly}周, B={B})")
    print(f"  日度 IID Bootstrap SE: 0.4168")
    print(f"  周度 MBB / 日度 IID 比率: {boot_se/0.4168:.2f}x")
else:
    print(f"  周度 MBB SE: {boot_se:.4f} (WarSpotting, b={b_weekly}周, B={B})")

# ===========================
# 3. Jackknife + BCa 校正
# ===========================
print(f"\n--- 3.1 Jackknife + BCa ---")

# Jackknife（每次删去一周）
jack_stats = np.zeros(n_weeks)
total_sum = np.sum(counts)
for i in range(n_weeks):
    jack_stats[i] = (total_sum - counts[i]) / (7 * (n_weeks - 1))

jack_mean = np.mean(jack_stats)

# BCa 偏差校正因子 z0
prop_less = np.mean(boot_rates < theta_hat)
prop_less = max(min(prop_less, 1 - 1/(2*B)), 1/(2*B))  # clamp
z0 = stats.norm.ppf(prop_less)

# BCa 加速常数 a_hat
diffs = jack_mean - jack_stats
numerator = np.sum(diffs ** 3)
denominator = 6 * (np.sum(diffs ** 2) ** 1.5)
a_hat = numerator / denominator if denominator > 1e-10 else 0.0

# BCa 修正分位点
def bca_quantile(prob):
    z = stats.norm.ppf(prob)
    denominator = 1 - a_hat * (z0 + z)
    if denominator <= 0:
        return 0.0 if prob < 0.5 else 1.0
    adj = stats.norm.cdf(z0 + (z0 + z) / denominator)
    return max(0.0, min(1.0, adj))

q_low = bca_quantile(ALPHA / 2)
q_high = bca_quantile(1 - ALPHA / 2)

bca_low = np.percentile(boot_rates, q_low * 100)
bca_high = np.percentile(boot_rates, q_high * 100)

print(f"偏差校正 z0: {z0:.4f}")
print(f"加速常数 a: {a_hat:.4f}")
print(f"BCa 修正分位点: [{q_low:.4f}, {q_high:.4f}]")
print(f"BCa 95% CI: [{bca_low:.4f}, {bca_high:.4f}]")

# ===========================
# 4. 分装备类型周度 Bootstrap
# ===========================
print(f"\n--- 4.1 分装备类型周度 Bootstrap ---")
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_by_type['week'] = daily_by_type.index.isocalendar().year.astype(str) + '-W' + \
                         daily_by_type.index.isocalendar().week.astype(str).str.zfill(2)

type_boot_results = []
major_types = ['Tanks', 'Infantry fighting vehicles', 'Self-propelled artillery',
               'Towed artillery', 'Drones', 'Transport']

for eq_type in major_types:
    if eq_type not in daily_by_type.columns:
        continue
    weekly_type = daily_by_type.groupby('week')[eq_type].sum()
    type_counts = weekly_type.values

    if np.sum(type_counts) < 50:
        continue

    theta_t = np.sum(type_counts) / (7 * len(type_counts))
    n_t = len(type_counts)

    # Moving Block Bootstrap (按周抽取块)
    b_t = max(choose_block_length_weekly(type_counts, max_lag=8), 1)
    bt = np.zeros(2000)
    k_t = int(np.ceil(n_t / max(b_t, 1)))
    for b_idx in range(2000):
        bs_weekly_t = np.zeros(n_t)
        pos = 0
        for _ in range(k_t):
            if n_t - max(b_t, 1) + 1 <= 0:
                start = 0
            else:
                start = np.random.randint(0, n_t - max(b_t, 1) + 1)
            block = type_counts[start:start + max(b_t, 1)]
            end = min(pos + len(block), n_t)
            bs_weekly_t[pos:end] = block[:end - pos]
            pos = end
            if pos >= n_t:
                break
        bt[b_idx] = np.sum(bs_weekly_t) / (7 * n_t)

    type_boot_results.append({
        'equipment_type': eq_type,
        'n_weeks': n_t,
        'total_losses': int(np.sum(type_counts)),
        'theta_hat_rate_per_day': round(theta_t, 4),
        'bootstrap_mean': round(np.mean(bt), 4),
        'bootstrap_se': round(np.std(bt, ddof=1), 4),
        'percentile_ci_low': round(np.percentile(bt, 2.5), 4),
        'percentile_ci_high': round(np.percentile(bt, 97.5), 4),
    })
    print(f"  {eq_type:<30s}: λ̂={theta_t:.3f}/天, SE={np.std(bt, ddof=1):.3f}")

# ===========================
# 5. 保存结果
# ===========================
print(f"\n--- 保存周度 Bootstrap 结果 ---")
os.makedirs(os.path.join('output', 'tables'), exist_ok=True)

# BCa 汇总
summary = pd.DataFrame([{
    'statistic': 'weekly_block_bootstrap_rate_per_day',
    'n_weeks': n_weeks,
    'bootstrap_reps': B,
    'seed': SEED,
    'theta_hat_rate_per_day': round(theta_hat, 6),
    'theta_hat_rate_per_30_days': round(theta_hat * 30, 6),
    'bootstrap_mean_rate_per_day': round(boot_mean, 6),
    'bootstrap_sd_rate_per_day': round(boot_se, 6),
    'percentile_ci_low_per_day': round(boot_percentile_ci[0], 6),
    'percentile_ci_high_per_day': round(boot_percentile_ci[1], 6),
    'bca_ci_low_per_day': round(bca_low, 6),
    'bca_ci_high_per_day': round(bca_high, 6),
    'bca_z0': round(z0, 6),
    'bca_acceleration': round(a_hat, 6),
    'bca_adjusted_quantile_low': round(q_low, 6),
    'bca_adjusted_quantile_high': round(q_high, 6),
}])
summary.to_csv(os.path.join('output', 'tables', 'weekly_bootstrap_bca.csv'), index=False)

# Bootstrap 分布
dist_df = pd.DataFrame({
    'replicate': range(1, B + 1),
    'rate_per_day': np.round(boot_rates, 6),
    'rate_per_30_days': np.round(boot_rates * 30, 6),
})
dist_df.to_csv(os.path.join('output', 'tables', 'weekly_bootstrap_distribution.csv'), index=False)

# Jackknife
jk_df = pd.DataFrame({
    'omitted_week': weekly['week'].values,
    'omitted_week_start': weekly['week_start'].values,
    'omitted_loss_count': counts,
    'jackknife_rate_per_day': np.round(jack_stats, 6),
})
jk_df.to_csv(os.path.join('output', 'tables', 'weekly_jackknife_rates.csv'), index=False)

# 分类型
type_df = pd.DataFrame(type_boot_results)
type_df.to_csv(os.path.join('output', 'tables', 'weekly_bootstrap_by_type.csv'), index=False)

print("✓ weekly_bootstrap_bca.csv")
print("✓ weekly_bootstrap_distribution.csv")
print("✓ weekly_jackknife_rates.csv")
print("✓ weekly_bootstrap_by_type.csv")
print(f"\n周度 Block Bootstrap 分析全部完成!")
print(f"\n核心结果对比:")
print(f"  周度 MBB BCa CI: [{bca_low:.4f}, {bca_high:.4f}] (WarSpotting数据, b={b_weekly}周, B={B})")
print(f"  注: 周度MBB保留了周间的时间依赖结构, 而非将各周视为独立")
print(f"  运行 python code/bootstrap.py 获取日度 Bootstrap 结果进行对比")
