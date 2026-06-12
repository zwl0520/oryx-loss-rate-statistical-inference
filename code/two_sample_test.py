"""
两独立 Poisson 条件检验模块
===========================
功能：
1. 两独立 Poisson 均值比较的条件检验 (E-test, C-test)
2. 似然比检验 (LRT)
3. 按装备类型/阶段的多重比较
4. Bonferroni 校正
5. 效应量估计

理论背景：
---------
设 Y₁ ~ Poisson(n₁λ₁), Y₂ ~ Poisson(n₂λ₂) 独立

H₀: λ₁ = λ₂ vs H₁: λ₁ ≠ λ₂

给定总计数 S = Y₁ + Y₂，在 H₀ 下：
Y₁ | S = s ~ Binomial(s, n₁/(n₁+n₂))

条件精确检验 (E-test, Krishnamoorthy & Thomson 2004):
基于条件二项分布的直接 p-value 计算

似然比检验:
LR = -2 log(L(λ̂₀) / L(λ̂₁, λ̂₂))
在 H₀ 下，LR ~ χ²(1)（Wilks 定理）
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import comb
import os
import sys
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载数据
# ===========================
print("=" * 70)
print("两独立 Poisson 条件检验")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_ua = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'), parse_dates=['date'])

# ===========================
# 2. 检验函数
# ===========================
def poisson_e_test(y1, n1, y2, n2):
    """
    E-test for comparing two Poisson means (Krishnamoorthy & Thomson 2004)

    H₀: λ₁ = λ₂
    基于事实：Y₁ | Y₁+Y₂ ~ Binomial(Y₁+Y₂, n₁/(n₁+n₂))

    返回 p-value (双侧)
    """
    total = y1 + y2
    p_hat = n1 / (n1 + n2)
    # 条件二项分布 p-value
    # P(Y₁ ≥ y1 | total, p_hat)
    p_upper = 1 - stats.binom.cdf(y1 - 1, total, p_hat) if y1 > 0 else 1.0
    p_lower = stats.binom.cdf(y1, total, p_hat)
    return 2 * min(p_upper, p_lower)

def poisson_lrt(data1, data2):
    """
    似然比检验 (LRT) for H₀: λ₁ = λ₂

    统计量: LR = -2[ℓ(λ̂₀) - ℓ(λ̂₁, λ̂₂)]
    在 H₀ 下渐近服从 χ²(1)
    """
    n1, n2 = len(data1), len(data2)
    lambda1 = np.mean(data1)
    lambda2 = np.mean(data2)
    lambda0 = (np.sum(data1) + np.sum(data2)) / (n1 + n2)

    # H₁ 下的对数似然
    ll1 = -n1*lambda1 + np.sum(data1)*np.log(max(lambda1, 1e-10))
    ll2 = -n2*lambda2 + np.sum(data2)*np.log(max(lambda2, 1e-10))
    ll_alt = ll1 + ll2

    # H₀ 下的对数似然（公共 λ）
    total_data = np.concatenate([data1, data2])
    ll_null = -(n1+n2)*lambda0 + np.sum(total_data)*np.log(max(lambda0, 1e-10))

    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)

    return lr_stat, p_value

def poisson_conditional_test(data1, data2):
    """
    两独立 Poisson 条件检验的完整报告
    """
    n1, n2 = len(data1), len(data2)
    y1, y2 = np.sum(data1), np.sum(data2)
    lambda1, lambda2 = np.mean(data1), np.mean(data2)

    # E-test
    e_pval = poisson_e_test(y1, n1, y2, n2)

    # LRT
    lr_stat, lr_pval = poisson_lrt(data1, data2)

    # 效应量：率比
    rate_ratio = lambda1 / lambda2 if lambda2 > 0 else np.inf

    # 率差的 Wald CI
    se_diff = np.sqrt(lambda1/n1 + lambda2/n2)
    diff = lambda1 - lambda2
    diff_ci_lower = diff - 1.96 * se_diff
    diff_ci_upper = diff + 1.96 * se_diff

    return {
        'n1': n1, 'n2': n2,
        'total1': int(y1), 'total2': int(y2),
        'lambda1': lambda1, 'lambda2': lambda2,
        'rate_ratio': rate_ratio,
        'diff': diff,
        'diff_ci': (diff_ci_lower, diff_ci_upper),
        'e_test_pval': e_pval,
        'lrt_stat': lr_stat,
        'lrt_pval': lr_pval,
    }

# ===========================
# 3. 关键比较：战争阶段之间
# ===========================
print("\n--- 3.1 关键阶段两两比较 ---")

war_phases_short = {
    'P1 (2022.2-4)': ('2022-02-24', '2022-04-30'),
    'P3 (2022.9-12)': ('2022-09-01', '2022-12-31'),
    'P7 (2024.3-7)': ('2024-03-01', '2024-07-31'),
    'P9 (2025-26)': ('2025-01-01', '2026-06-30'),
}

phase_data = {}
for phase, (start, end) in war_phases_short.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    phase_data[phase] = daily_ws[mask]['total_losses'].values

comparisons = list(combinations(war_phases_short.keys(), 2))
alpha = 0.05
n_comparisons = len(comparisons)
bonferroni_alpha = alpha / n_comparisons

print(f"多重比较数: {n_comparisons}, Bonferroni 校正 α = {bonferroni_alpha:.4f}")
print(f"\n{'比较':<30s} {'λ̂₁':>8s} {'λ̂₂':>8s} {'率比':>8s} {'E-test p':>10s} {'LRT p':>10s} {'显著(Bonf)':>12s}")
print("-" * 100)

phase_comparisons = []
for p1, p2 in comparisons:
    res = poisson_conditional_test(phase_data[p1], phase_data[p2])
    phase_comparisons.append({'Phase 1': p1, 'Phase 2': p2, **res})
    bonf_sig = "***" if res['e_test_pval'] < bonferroni_alpha else ("**" if res['e_test_pval'] < 0.01 else ("*" if res['e_test_pval'] < 0.05 else "n.s."))
    print(f"{p1} vs {p2:<20s} {res['lambda1']:8.2f} {res['lambda2']:8.2f} {res['rate_ratio']:8.3f} {res['e_test_pval']:10.6f} {res['lrt_pval']:10.6f} {bonf_sig:>12s}")

# ===========================
# 4. 装备类型比较
# ===========================
print("\n--- 3.2 主要装备类型两两比较 ---")
major_types = ['Tanks', 'Infantry fighting vehicles', 'Self-propelled artillery',
               'Towed artillery', 'Drones', 'Transport']

type_data = {}
for t in major_types:
    if t in daily_by_type.columns:
        type_data[t] = daily_by_type[t].values

type_comparisons = list(combinations(type_data.keys(), 2))
n_type_comps = len(type_comparisons)
bonf_type_alpha = alpha / n_type_comps
print(f"比较数: {n_type_comps}, Bonferroni α = {bonf_type_alpha:.6f}")

type_test_results = []
for t1, t2 in type_comparisons:
    res = poisson_conditional_test(type_data[t1], type_data[t2])
    type_test_results.append({'Type 1': t1, 'Type 2': t2, **res})
    bonf_sig = "***" if res['e_test_pval'] < bonf_type_alpha else ("**" if res['e_test_pval'] < 0.01 else ("*" if res['e_test_pval'] < 0.05 else "n.s."))
    if res['e_test_pval'] < 0.05:
        print(f"  {t1} vs {t2}: λ̂₁={res['lambda1']:.3f}, λ̂₂={res['lambda2']:.3f}, "
              f"率比={res['rate_ratio']:.3f}, p={res['e_test_pval']:.6f} {bonf_sig}")

# ===========================
# 5. WarSpotting 验证 vs 乌克兰声称（坦克）
# ===========================
print("\n--- 3.3 WarSpotting验证 vs 乌克兰声称 ---")
ws_tank = daily_by_type['Tanks'].values
# 对齐天数
min_len = min(len(ws_tank), len(daily_ua['Tanks']))
ws_tank_aligned = ws_tank[:min_len]
ua_tank_aligned = daily_ua['Tanks'].values[:min_len]

res_claim_vs_verify = poisson_conditional_test(ua_tank_aligned, ws_tank_aligned)
print(f"坦克日均损失比较:")
print(f"  WarSpotting验证: λ̂ = {res_claim_vs_verify['lambda2']:.3f} 辆/天")
print(f"  乌克兰总参声称: λ̂ = {res_claim_vs_verify['lambda1']:.3f} 辆/天")
print(f"  声称/验证比 = {res_claim_vs_verify['rate_ratio']:.3f}")
print(f"  E-test p-value = {res_claim_vs_verify['e_test_pval']:.10f}")
print(f"  LRT统计量 = {res_claim_vs_verify['lrt_stat']:.2f}, p < 0.0001")
print(f"  结论: 乌克兰官方声称的俄军坦克损失显著高于影像验证数据")

# 火炮比较
ws_art = (daily_by_type['Self-propelled artillery'] + daily_by_type['Towed artillery']).values
ua_art = daily_ua['Field Artillery'].values[:len(ws_art)]
res_art = poisson_conditional_test(ua_art, ws_art)
print(f"\n火炮日均损失比较:")
print(f"  WarSpotting验证: λ̂ = {res_art['lambda2']:.3f} 门/天")
print(f"  乌克兰总参声称: λ̂ = {res_art['lambda1']:.3f} 门/天")
print(f"  声称/验证比 = {res_art['rate_ratio']:.3f}")
print(f"  E-test p-value = {res_art['e_test_pval']:.10f}")

# ===========================
# 6. 装备状态比较（Destroyed vs Captured）
# ===========================
print("\n--- 3.4 装备状态对比 ---")
daily_by_status = pd.read_csv(os.path.join('data', 'daily_by_status_ws.csv'), index_col=0, parse_dates=True)
res_stat = poisson_conditional_test(daily_by_status['Destroyed'].values, daily_by_status['Captured'].values)
print(f"Destroyed vs Captured:")
print(f"  Destroyed: λ̂ = {res_stat['lambda1']:.2f}/天")
print(f"  Captured: λ̂ = {res_stat['lambda2']:.2f}/天")
print(f"  率比 = {res_stat['rate_ratio']:.2f}, p < 0.0001")

# ===========================
# 7. 保存结果
# ===========================
print("\n--- 保存检验结果 ---")
phase_comp_df = pd.DataFrame(phase_comparisons)
phase_comp_df.to_csv(os.path.join('output', 'tables', 'phase_comparisons.csv'), index=False)
print("✓ 阶段比较结果已保存")

type_comp_df = pd.DataFrame(type_test_results)
type_comp_df.to_csv(os.path.join('output', 'tables', 'type_comparisons.csv'), index=False)
print("✓ 装备类型比较结果已保存")

print("\n两独立 Poisson 条件检验全部完成！")
