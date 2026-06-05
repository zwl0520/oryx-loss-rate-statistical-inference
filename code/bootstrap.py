"""
Block Bootstrap 模块
===================
功能：
1. 块长度选择（基于自相关分析）
2. Moving Block Bootstrap (MBB) 重抽样
3. BCa 偏差校正加速置信区间
4. Bootstrap 与参数 CI 的比较
5. Bootstrap 分布可视化

理论背景：
---------
时间序列数据存在自相关，独立同分布 Bootstrap 会低估方差。
Block Bootstrap 通过保留块内的时间结构来保持依赖性。

BCa (Bias-Corrected and Accelerated) CI:
- 偏差校正：调整 Bootstrap 分布的中心偏移
- 加速：调整方差的非恒定性（偏度校正）
- 分位点: α₁ = Φ(ẑ₀ + (ẑ₀ + z_α)/(1 - â(ẑ₀ + z_α)))
           α₂ = Φ(ẑ₀ + (ẑ₀ + z_{1-α})/(1 - â(ẑ₀ + z_{1-α})))
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

# ===========================
# 1. 加载数据
# ===========================
print("=" * 70)
print("Block Bootstrap 分析")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_ua = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'), parse_dates=['date'])

# ===========================
# 2. 块长度选择
# ===========================
def choose_block_length(data, max_lag=30):
    """
    基于自相关函数选择块长度。

    方法：找到自相关衰减到 2/√n 以下的第一个滞后阶数。
    对于强相关数据，使用更保守的选择。
    """
    n = len(data)
    acf = []
    x = data - np.mean(data)
    var = np.var(x) * n  # 未标准化

    if var < 1e-10:
        return 1

    for lag in range(1, min(max_lag, n // 4)):
        acf_val = np.sum(x[lag:] * x[:-lag]) / var
        acf.append(acf_val)

    acf = np.array(acf)
    # 找到 |ACF| < 2/√n 的第一个滞后
    threshold = 2 / np.sqrt(n)
    below_threshold = np.where(np.abs(acf) < threshold)[0]

    if len(below_threshold) > 0:
        block_len = max(below_threshold[0] + 1, 3)
    else:
        # 默认：n^(1/3) - 常见经验公式
        block_len = max(int(n ** (1/3)), 3)

    return min(block_len, max_lag)

print("\n--- 2.1 块长度选择 ---")
data = daily_ws['total_losses'].values
b_opt = choose_block_length(data)
print(f"数据长度: n = {len(data)}")
print(f"推荐块长度: b = {b_opt}")

# 显示前几个自相关系数
x = data - np.mean(data)
var = np.sum(x**2)
acf_vals = []
for lag in range(1, 11):
    acf = np.sum(x[lag:] * x[:-lag]) / var
    acf_vals.append(acf)
print(f"自相关 (lag 1-10): {[f'{a:.3f}' for a in acf_vals]}")

# ===========================
# 3. Block Bootstrap 实现
# ===========================
def block_bootstrap(data, B=2000, block_len=None):
    """
    Moving Block Bootstrap (MBB)

    步骤：
    1. 将数据分成 n-b+1 个重叠的长度为 b 的块
    2. 有放回地抽取 k = n/b 个块（向上取整）
    3. 连接所有块形成长度为 n 的 Bootstrap 样本
    4. 计算每个样本的均值
    """
    n = len(data)
    if block_len is None:
        block_len = choose_block_length(data)

    # 计算需要的块数
    k = int(np.ceil(n / block_len))

    bootstrap_means = np.zeros(B)

    for i in range(B):
        # 随机选取 k 个块的起始位置
        block_starts = np.random.randint(0, n - block_len + 1, k)
        bs_sample = []
        for start in block_starts:
            bs_sample.extend(data[start:start + block_len])
        bs_sample = np.array(bs_sample[:n])  # 截断至原长度
        bootstrap_means[i] = np.mean(bs_sample)

    return bootstrap_means

def jackknife_means(data):
    """Jackknife 均值"""
    n = len(data)
    jk_means = np.zeros(n)
    total = np.sum(data)
    for i in range(n):
        jk_means[i] = (total - data[i]) / (n - 1)
    return jk_means

def bca_ci(data, B=2000, alpha=0.05, block_len=None):
    """
    BCa (Bias-Corrected and Accelerated) Bootstrap 置信区间

    步骤：
    1. 偏差校正 ẑ₀ = Φ⁻¹(#{θ̂* < θ̂} / B)
    2. 加速 â = ∑(θ̄ - θ̂_{(i)})³ / (6 · (∑(θ̄ - θ̂_{(i)})²)^{3/2})
       (通过 Jackknife 估计)
    3. 计算修正分位点
    4. 取 Bootstrap 分布的修正分位点作为 CI 边界
    """
    n = len(data)
    theta_hat = np.mean(data)

    # Bootstrap
    boot_means = block_bootstrap(data, B, block_len)

    # 1. 偏差校正
    z0 = stats.norm.ppf(np.mean(boot_means < theta_hat))

    # 2. 加速常数（通过 Jackknife）
    jk_means = jackknife_means(data)
    jk_mean_overall = np.mean(jk_means)
    numerator = np.sum((jk_mean_overall - jk_means) ** 3)
    denominator = 6 * (np.sum((jk_mean_overall - jk_means) ** 2)) ** 1.5
    if denominator > 1e-10:
        a_hat = numerator / denominator
    else:
        a_hat = 0.0

    # 3. 修正分位点
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)

    alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a_hat * (z0 + z_1_alpha)))

    # 4. Bootstrap 分位点 (BCa 修正)
    lower = np.percentile(boot_means, alpha1 * 100)
    upper = np.percentile(boot_means, alpha2 * 100)

    return {
        'lower': lower,
        'upper': upper,
        'z0': z0,
        'a_hat': a_hat,
        'boot_means': boot_means,
        'boot_mean': np.mean(boot_means),
        'boot_se': np.std(boot_means, ddof=1),
        'percentile_ci': (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)),
    }

# ===========================
# 4. 对总体数据应用 Bootstrap
# ===========================
print("\n--- 4.1 总体 Bootstrap CI ---")
ws_data = daily_ws['total_losses'].values
n = len(ws_data)
theta_hat = np.mean(ws_data)
print(f"MLE λ̂ = {theta_hat:.4f}")

# 先运行 iid Bootstrap 作为基准
print("运行 IID Bootstrap (B=2000)...")
iid_boot = np.zeros(2000)
for i in range(2000):
    iid_boot[i] = np.mean(np.random.choice(ws_data, size=n, replace=True))
iid_pct_ci = (np.percentile(iid_boot, 2.5), np.percentile(iid_boot, 97.5))
print(f"  IID Bootstrap 95% CI: [{iid_pct_ci[0]:.4f}, {iid_pct_ci[1]:.4f}]")

# Block Bootstrap
print(f"运行 Block Bootstrap (b={b_opt}, B=2000)...")
bca_result = bca_ci(ws_data, B=2000, block_len=b_opt)
print(f"  Block Bootstrap 均值: {bca_result['boot_mean']:.4f}")
print(f"  Block Bootstrap SE: {bca_result['boot_se']:.4f}")
print(f"  偏差校正 ẑ₀ = {bca_result['z0']:.4f}")
print(f"  加速常数 â = {bca_result['a_hat']:.4f}")
print(f"  Percentile 95% CI: [{bca_result['percentile_ci'][0]:.4f}, {bca_result['percentile_ci'][1]:.4f}]")
print(f"  BCa 95% CI: [{bca_result['lower']:.4f}, {bca_result['upper']:.4f}]")

# ===========================
# 5. 分阶段 Bootstrap
# ===========================
print("\n--- 4.2 关键阶段 Bootstrap CI ---")
war_phases_short = {
    'Phase 1 (Initial Invasion)': ('2022-02-24', '2022-04-30'),
    'Phase 3 (Counteroffensive)': ('2022-09-01', '2022-12-31'),
    'Phase 7 (Spring 2024)': ('2024-03-01', '2024-07-31'),
    'Phase 9 (Recent)': ('2025-01-01', '2026-06-30'),
}

phase_bs_results = []
for phase, (start, end) in war_phases_short.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    data_p = daily_ws[mask]['total_losses'].values
    if len(data_p) < 20:
        continue
    b = choose_block_length(data_p, max_lag=15)
    res = bca_ci(data_p, B=2000, block_len=b)
    phase_bs_results.append({
        'Phase': phase,
        'MLE': np.mean(data_p),
        'BCa_Lower': res['lower'],
        'BCa_Upper': res['upper'],
        'Bootstrap_SE': res['boot_se'],
        'z0': res['z0'],
        'a_hat': res['a_hat'],
    })
    print(f"  {phase}: MLE={np.mean(data_p):.2f}, BCa 95% CI=[{res['lower']:.2f}, {res['upper']:.2f}], SE={res['boot_se']:.2f}")

# ===========================
# 6. 对比 IID vs Block Bootstrap
# ===========================
print("\n--- 4.3 IID vs Block Bootstrap 对比 ---")
print("使用不同块长度评估对标准误估计的影响")

block_lengths = [1, 3, 5, 7, 10, 14, 20, 30]
for bl in block_lengths:
    if bl <= 1:
        boot_means = iid_boot  # 复用已计算的 IID
    else:
        boot_means = block_bootstrap(ws_data, B=2000, block_len=bl)
    boot_se = np.std(boot_means, ddof=1)
    print(f"  块长度 b={bl:2d}: Bootstrap SE = {boot_se:.4f}")

# ===========================
# 7. 两样本 Bootstrap 检验（阶段对比）
# ===========================
print("\n--- 4.4 Bootstrap 两阶段差异检验 ---")
# 对比 Phase 1 和 Phase 9
p1_mask = (daily_ws['date'] >= '2022-02-24') & (daily_ws['date'] <= '2022-04-30')
p9_mask = (daily_ws['date'] >= '2025-01-01') & (daily_ws['date'] <= '2026-06-03')

p1_data = daily_ws[p1_mask]['total_losses'].values
p9_data = daily_ws[p9_mask]['total_losses'].values

# 观测差异
obs_diff = np.mean(p1_data) - np.mean(p9_data)
print(f"Phase 1 均值: {np.mean(p1_data):.2f}")
print(f"Phase 9 均值: {np.mean(p9_data):.2f}")
print(f"观测差异: {obs_diff:.2f}")

# Bootstrap 差异分布
B = 2000
diff_boot = np.zeros(B)
b1 = choose_block_length(p1_data, max_lag=15)
b9 = choose_block_length(p9_data, max_lag=15)
for i in range(B):
    bs1 = block_bootstrap(p1_data, B=1, block_len=b1)[0]
    bs9 = block_bootstrap(p9_data, B=1, block_len=b9)[0]
    diff_boot[i] = bs1 - bs9

# Bootstrap CI for difference
diff_ci = np.percentile(diff_boot, [2.5, 97.5])
# Bootstrap p-value (two-sided)
p_val = 2 * min(np.mean(diff_boot <= 0), np.mean(diff_boot >= 0))
print(f"Bootstrap 95% CI for difference: [{diff_ci[0]:.2f}, {diff_ci[1]:.2f}]")
print(f"Bootstrap p-value: {p_val:.4f}")

# ===========================
# 8. 保存结果
# ===========================
print("\n--- 保存 Bootstrap 结果 ---")
bs_summary = pd.DataFrame(phase_bs_results)
bs_summary.to_csv(os.path.join('output', 'tables', 'bootstrap_results.csv'), index=False)

# 保存 Bootstrap 分布
boot_dist_df = pd.DataFrame({
    'IID_Bootstrap_Means': iid_boot,
    'Block_Bootstrap_Means': bca_result['boot_means']
})
boot_dist_df.to_csv(os.path.join('output', 'tables', 'bootstrap_distributions.csv'), index=False)

print("✓ Bootstrap 结果已保存")
print("\nBlock Bootstrap 分析全部完成！")
