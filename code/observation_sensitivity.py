"""
观测概率灵敏度分析模块
=====================
对 Oryx 影像验证数据的"被观测概率"进行敏感性分析。

核心问题：如果俄方和乌方装备损失的"被观测概率"不同，真实的双方损失比率
        会与观测到的 2.07:1 有多大偏差？

方法：
1. 构建观测概率网格 (p_RU, p_UA) ∈ [0.4, 1.0] × [0.4, 1.0]
2. 对每对 (p_RU, p_UA) 计算修正后的双方损失数和率比
3. 识别率比 ≤ 1（即修正后乌方 ≥ 俄方）的观测概率组合
4. 进行总体和分装备类型的灵敏度分析
"""

import pandas as pd
import numpy as np
import os, sys

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载数据
# ===========================
print("=" * 70)
print("观测概率灵敏度分析")
print("=" * 70)

# 使用整理后的 Oryx 双方数据（item_count 求和，更精确的逐件计数）
# 注: 此数据与 both_sides_analysis.py 使用的 oryx_both_sides.csv 是不同来源
#      oryx_losses_combined.csv: 直接抓取 Oryx HTML, 使用 item_count 加权求和
#      oryx_both_sides.csv: 来自 leedrake5 Google Sheet 每日累计
#      两者总数略有差异（约2%），但不影响灵敏度分析的结论
try:
    df_oryx = pd.read_csv(os.path.join('data', 'oryx_losses_combined.csv'))
    ru_total = int(df_oryx[df_oryx['side'] == 'Russia']['item_count'].sum())
    ua_total = int(df_oryx[df_oryx['side'] == 'Ukraine']['item_count'].sum())
    print(f"Oryx 逐件数据 (item_count): 俄方 {ru_total} 件, 乌方 {ua_total} 件")
    # 同时加载日度数据用于对比
    try:
        daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
        ru_daily = int(daily['Russia_Total'].sum())
        ua_daily = int(daily['Ukraine_Total'].sum())
        if ru_daily != ru_total or ua_daily != ua_total:
            print(f"  日度聚合数据:             俄方 {ru_daily} 件, 乌方 {ua_daily} 件")
            print(f"  差异: RU {ru_daily-ru_total:+d}, UA {ua_daily-ua_total:+d} (不同数据源的计数口径差异)")
    except:
        pass
except:
    # 回退到双方日度数据
    daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
    ru_total = int(daily['Russia_Total'].sum())
    ua_total = int(daily['Ukraine_Total'].sum())
    print(f"日度聚合数据: 俄方 {ru_total} 件, 乌方 {ua_total} 件")

observed_ratio = ru_total / ua_total
print(f"观测率比 RU/UA: {observed_ratio:.3f}")

# ===========================
# 2. 灵敏度网格构建
# ===========================
print("\n--- 2.1 总体灵敏度网格 ---")
obs_probs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for p_ru in obs_probs:
    for p_ua in obs_probs:
        corrected_ru = ru_total / p_ru
        corrected_ua = ua_total / p_ua
        corrected_ratio = corrected_ru / corrected_ua
        results.append({
            'p_RU': p_ru,
            'p_UA': p_ua,
            'observed_RU': ru_total,
            'observed_UA': ua_total,
            'corrected_RU': round(corrected_ru, 1),
            'corrected_UA': round(corrected_ua, 1),
            'observed_ratio': round(observed_ratio, 3),
            'corrected_ratio': round(corrected_ratio, 3),
            'ratio_still_above_1': corrected_ratio > 1.0,
        })

sensitivity_df = pd.DataFrame(results)

# ===========================
# 3. 关键敏感性发现
# ===========================
print("\n--- 3.1 关键场景 ---")

# 场景1: 同等观测概率
print("\n场景1: 双方观测率相同 (p_RU = p_UA)")
for p in obs_probs:
    row = sensitivity_df[(sensitivity_df['p_RU'] == p) & (sensitivity_df['p_UA'] == p)]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  p={p:.1f}: 修正后 RU={r['corrected_RU']:.0f}, UA={r['corrected_UA']:.0f}, "
              f"率比={r['corrected_ratio']:.3f} (不变)")

# 场景2: 乌方观测率低于俄方（俄占区影像少）
print("\n场景2: 乌方观测率低于俄方（乌方在俄占区的损失更少被拍到）")
for p_ru in [0.6, 0.7, 0.8]:
    for p_ua in [0.4, 0.5]:
        row = sensitivity_df[(sensitivity_df['p_RU'] == p_ru) & (sensitivity_df['p_UA'] == p_ua)]
        if len(row) > 0:
            r = row.iloc[0]
            marker = " ← 率比逆转!" if not r['ratio_still_above_1'] else ""
            print(f"  p_RU={p_ru:.1f}, p_UA={p_ua:.1f}: 修正率比={r['corrected_ratio']:.3f}{marker}")

# 场景3: 逆转边界——找到使率比 ≤ 1 的观测概率组合
print("\n场景3: 使率比逆转 (RU/UA ≤ 1.0) 的观测概率组合")
reversal_cases = sensitivity_df[~sensitivity_df['ratio_still_above_1']]
if len(reversal_cases) > 0:
    for _, r in reversal_cases.iterrows():
        print(f"  p_RU={r['p_RU']:.1f}, p_UA={r['p_UA']:.1f} → 修正率比={r['corrected_ratio']:.3f}")
else:
    print("  在 [0.4, 1.0] × [0.4, 1.0] 网格内未找到逆转组合")
    # 计算需要多大的观测概率差才能逆转
    print(f"  要使率比≤1，需要 p_UA/p_RU ≤ {1/observed_ratio:.3f}")
    print(f"  例如: p_RU=0.9时, p_UA ≤ {0.9/observed_ratio:.3f} 才能使乌方修正损失≥俄方")

# ===========================
# 4. 按装备类型的灵敏度
# ===========================
print("\n--- 4.1 分装备类型灵敏度 ---")
try:
    # 使用 Oryx 合并数据的分类型统计
    cat_stats = df_oryx.groupby(['side', 'category'])['item_count'].sum().unstack(fill_value=0)
    ru_by_cat = cat_stats.loc['Russia'] if 'Russia' in cat_stats.index else None
    ua_by_cat = cat_stats.loc['Ukraine'] if 'Ukraine' in cat_stats.index else None

    if ru_by_cat is not None and ua_by_cat is not None:
        common_cats = ru_by_cat.index.intersection(ua_by_cat.index)
        print(f"  {'装备类型':<35s} {'RU观测':>6s} {'UA观测':>6s} {'观测率比':>8s} {'逆转需p_UA≤':>10s}")
        print(f"  {'-'*70}")

        for cat in sorted(common_cats, key=lambda c: ru_by_cat[c] / max(ua_by_cat[c], 1), reverse=True):
            ru_c = int(ru_by_cat[cat])
            ua_c = int(ua_by_cat[cat])
            if ru_c + ua_c < 20:
                continue
            cat_ratio = ru_c / max(ua_c, 1)
            # 在p_RU=0.8假设下，需p_UA多低才能逆转
            p_ua_needed = 0.8 / cat_ratio if cat_ratio > 0.8 else 0.8
            print(f"  {cat:<35s} {ru_c:6d} {ua_c:6d} {cat_ratio:8.2f} {p_ua_needed:10.3f}")
except Exception as e:
    print(f"  分类型分析出错: {e}")

# ===========================
# 5. 关键结论
# ===========================
print(f"\n{'='*70}")
print("灵敏度分析关键结论:")
print(f"  1. 观测率比 RU/UA = {observed_ratio:.2f}")
print(f"  2. 若双方观测概率相同，修正不影响率比")
print(f"  3. 乌方观测概率需比俄方低超过 {1 - 1/observed_ratio:.0%} 才能使真实率比≤1")
print(f"  4. 考虑到俄占区影像获取难度更大，真实率比可能低于 {observed_ratio:.2f}")
print(f"  5. 最极端假设下 (p_RU=1.0, p_UA=0.4)，修正率比 = {observed_ratio * 0.4:.2f}")

# ===========================
# 6. 保存
# ===========================
print(f"\n--- 保存结果 ---")
os.makedirs(os.path.join('output', 'tables'), exist_ok=True)
sensitivity_df.to_csv(os.path.join('output', 'tables', 'observation_sensitivity_overall.csv'), index=False)
print("✓ observation_sensitivity_overall.csv")
print("\n观测概率灵敏度分析全部完成！")
