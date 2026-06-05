"""
Poisson 置信区间模块
===================
功能：
1. Wald 置信区间（渐近正态近似）
2. Score 置信区间（Wilson型，基于得分统计量）
3. Exact 置信区间（Garwood 1936，基于 χ² 分布精确构造）
4. 三种方法的覆盖概率模拟比较
5. 可视化对比

理论背景：
---------
设 Y₁,...,Yₙ ~ i.i.d. Poisson(λ) 或 Y ~ Poisson(nλ)（对总计数而言）

1. Wald CI (Asymptotic Normal):
   λ̂ ± z_{α/2} * √(λ̂/n)
   缺点：对小样本或小λ覆盖率不足，可能产生负下限

2. Score CI (Wilson-type):
   解 (λ̂ - λ)² / (λ/n) = z²_{α/2}
   得到 λ = λ̂ + z²/(2n) ± z * √(λ̂/n + z²/(4n²))
   优点：比 Wald 更稳健，不会产生负下限

3. Exact CI (Garwood 1936):
   基于 χ² 分布的关系：
   下限: ½ χ²_{2Y, α/2} / n  (当 Y>0)
   上限: ½ χ²_{2(Y+1), 1-α/2} / n
   下限为 0 (当 Y=0)
   优点：保证覆盖率 ≥ 名义水平（保守）
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载数据
# ===========================
print("=" * 70)
print("Poisson 置信区间分析与比较")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)

# ===========================
# 2. 三种置信区间函数
# ===========================
def wald_ci(data, alpha=0.05):
    """Wald 置信区间：λ̂ ± z_{α/2} * √(λ̂/n)"""
    n = len(data)
    lambda_hat = np.mean(data)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(lambda_hat / n)
    lower = max(0, lambda_hat - z * se)
    upper = lambda_hat + z * se
    return lower, upper, lambda_hat, se

def score_ci(data, alpha=0.05):
    """Score CI (Wilson-type for Poisson)

    解方程 (λ̂ - λ)² / (λ/n) = z² 得到：
    λ = λ̂ + z²/(2n) ± z * √(λ̂/n + z²/(4n²))
    """
    n = len(data)
    lambda_hat = np.mean(data)
    z = stats.norm.ppf(1 - alpha / 2)
    bias = z**2 / (2 * n)
    center = lambda_hat + bias
    margin = z * np.sqrt(lambda_hat / n + z**2 / (4 * n**2))
    lower = max(0, center - margin)
    upper = center + margin
    return lower, upper, center, margin

def exact_ci(data, alpha=0.05):
    """Exact (Garwood) CI: 基于 χ² 分布

    下限: χ²_{2Y, α/2} / (2n)
    上限: χ²_{2(Y+1), 1-α/2} / (2n)
    其中 Y = ∑x_i 为总计数
    """
    n = len(data)
    Y = np.sum(data)
    if Y == 0:
        lower = 0.0
    else:
        lower = stats.chi2.ppf(alpha / 2, 2 * Y) / (2 * n)
    upper = stats.chi2.ppf(1 - alpha / 2, 2 * (Y + 1)) / (2 * n)
    return lower, upper

def all_cis(data, alpha=0.05, label=""):
    """计算所有三种 CI 并打印"""
    w_l, w_u, lhat, se = wald_ci(data, alpha)
    s_l, s_u, _, _ = score_ci(data, alpha)
    e_l, e_u = exact_ci(data, alpha)

    print(f"\n{label}")
    print(f"  样本量(天): {len(data)}, 总计数: {np.sum(data):.0f}")
    print(f"  MLE λ̂ = {lhat:.4f}, SE = {se:.4f}")
    print(f"  Wald 95% CI:    [{w_l:.4f}, {w_u:.4f}]  宽度={w_u-w_l:.4f}")
    print(f"  Score 95% CI:   [{s_l:.4f}, {s_u:.4f}]  宽度={s_u-s_l:.4f}")
    print(f"  Exact 95% CI:   [{e_l:.4f}, {e_u:.4f}]  宽度={e_u-e_l:.4f}")
    return (w_l, w_u), (s_l, s_u), (e_l, e_u)

# ===========================
# 3. 总体与分阶段 CI
# ===========================
print("\n--- 3.1 总体日均损失率置信区间 ---")
all_cis(daily_ws['total_losses'].values, label="WarSpotting 总体 (全时段)")

# 分阶段
war_phases = {
    'Phase 1: Initial Invasion': ('2022-02-24', '2022-04-30'),
    'Phase 3: Counteroffensive': ('2022-09-01', '2022-12-31'),
    'Phase 9: Recent': ('2025-01-01', '2026-06-30'),
}

print("\n--- 3.2 关键阶段 CI 比较 ---")
for phase, (start, end) in war_phases.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    all_cis(daily_ws[mask]['total_losses'].values, label=phase)

# ===========================
# 4. 按装备类型 CI
# ===========================
print("\n--- 3.3 主要装备类型 CI 比较 ---")
major_types = ['Tanks', 'Infantry fighting vehicles', 'Self-propelled artillery',
               'Towed artillery', 'Drones']
for eq_type in major_types:
    if eq_type in daily_by_type.columns:
        all_cis(daily_by_type[eq_type].values, label=eq_type)

# ===========================
# 5. 蒙特卡洛模拟：CI 覆盖率比较
# ===========================
print("\n--- 3.4 覆盖率 Monte Carlo 模拟 ---")
print("模拟设定: Poisson(λ), n=30天, N_sim=10000")
print("评估各CI方法在名义95%水平下的实际覆盖率")

np.random.seed(42)

def simulate_coverage(true_lambda, n_days=30, n_sim=10000, alpha=0.05):
    """模拟三种 CI 方法的覆盖率"""
    wald_cover = 0
    score_cover = 0
    exact_cover = 0

    for _ in range(n_sim):
        data = np.random.poisson(true_lambda, n_days)

        # Wald
        w_l, w_u, _, _ = wald_ci(data, alpha)
        if w_l <= true_lambda <= w_u:
            wald_cover += 1

        # Score
        s_l, s_u, _, _ = score_ci(data, alpha)
        if s_l <= true_lambda <= s_u:
            score_cover += 1

        # Exact
        e_l, e_u = exact_ci(data, alpha)
        if e_l <= true_lambda <= e_u:
            exact_cover += 1

    return (wald_cover / n_sim, score_cover / n_sim, exact_cover / n_sim)

lambda_values = [0.5, 1, 2, 5, 10, 15, 20, 50]
sim_results = []
for lam in lambda_values:
    w, s, e = simulate_coverage(lam, n_days=30, n_sim=10000)
    sim_results.append({'lambda': lam, 'Wald': w, 'Score': s, 'Exact': e})
    print(f"  λ={lam:5.1f}: Wald={w:.3f}, Score={s:.3f}, Exact={e:.3f}")

# 保存模拟结果
sim_df = pd.DataFrame(sim_results)
sim_df.to_csv(os.path.join('output', 'tables', 'ci_coverage_simulation.csv'), index=False)
print("\n✓ 覆盖率模拟结果已保存")

# ===========================
# 6. 样本量对 CI 宽度的影响
# ===========================
print("\n--- 3.5 样本量对CI宽度的影响 ---")
true_lambda = 15.0  # 约等于总体日均损失
n_values = [7, 14, 30, 60, 90, 180, 365]
print(f"λ_true = {true_lambda}")
for n in n_values:
    data = np.random.poisson(true_lambda, n)
    w_l, w_u, _, _ = wald_ci(data)
    e_l, e_u = exact_ci(data)
    print(f"  n={n:4d}: Wald宽度={w_u-w_l:.3f}, Exact宽度={e_u-e_l:.3f}")

# ===========================
# 7. 生成置信区间比较数据（供可视化使用）
# ===========================
print("\n--- 3.6 保存CI比较数据 ---")
ci_comparison = []

# 总体
for name, data in [('Overall', daily_ws['total_losses'].values)] + \
                   [(t, daily_by_type[t].values) for t in major_types if t in daily_by_type.columns]:
    w_l, w_u, lhat, se = wald_ci(data)
    s_l, s_u, _, _ = score_ci(data)
    e_l, e_u = exact_ci(data)
    ci_comparison.append({
        'Category': name,
        'MLE': lhat,
        'Wald_Lower': w_l, 'Wald_Upper': w_u,
        'Score_Lower': s_l, 'Score_Upper': s_u,
        'Exact_Lower': e_l, 'Exact_Upper': e_u,
        'SE': se
    })

ci_df = pd.DataFrame(ci_comparison)
ci_df.to_csv(os.path.join('output', 'tables', 'ci_comparison.csv'), index=False)
print("✓ CI 比较数据已保存至 output/tables/ci_comparison.csv")
print("\n置信区间分析全部完成！")
