"""
观测概率偏误校正与灵敏度分析模块
===============================
对 Oryx 影像验证数据的"被观测概率"进行统计偏误校正。

选题 T1 核心方法链的最后一环：
  MLE → CI(Wald/Score/Exact) → Block Bootstrap(BCa) → E-test
  → 分层估计 → ★ 观测偏误校正与稳健推断

核心统计问题：
  影像验证损失 Y_obs 与真实损失 Y_true 满足 Y_obs = p × Y_true
  其中 p ∈ (0,1] 为"被观测概率"，因作战地域和装备类型而异。
  对于双方率比 ρ = λ_RU/λ_UA，有：
    ρ_true = ρ_obs × (p_UA / p_RU)

方法：
  1. 构建观测概率网格 → 偏误校正后的率比范围
  2. 利用官方声称数据约束观测概率下界（p ≥ 验证数/声称数）
  3. 基于 Bootstrap 的偏误校正：从合理先验中抽样 p 值，
     获得偏误校正后率比的完整抽样分布和置信区间
  4. 报告 P(ρ_corrected > 1) 作为结论稳健性的量化指标
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, sys, random

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载数据
# ===========================
print("=" * 70)
print("观测概率偏误校正与灵敏度分析")
print("=" * 70)

# 1a. Oryx 双方数据（item_count 加权）
try:
    df_oryx = pd.read_csv(os.path.join('data', 'oryx_losses_combined.csv'))
    ru_total = int(df_oryx[df_oryx['side'] == 'Russia']['item_count'].sum())
    ua_total = int(df_oryx[df_oryx['side'] == 'Ukraine']['item_count'].sum())
    print(f"Oryx 逐件数据 (item_count): 俄方 {ru_total} 件, 乌方 {ua_total} 件")
    try:
        daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
        ru_daily = int(daily['Russia_Total'].sum())
        ua_daily = int(daily['Ukraine_Total'].sum())
        if ru_daily != ru_total or ua_daily != ua_total:
            print(f"  日度聚合数据:             俄方 {ru_daily} 件, 乌方 {ua_daily} 件")
            print(f"  差异: RU {ru_daily-ru_total:+d}, UA {ua_daily-ua_total:+d} (不同数据源口径差异)")
    except:
        pass
except:
    daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
    ru_total = int(daily['Russia_Total'].sum())
    ua_total = int(daily['Ukraine_Total'].sum())
    print(f"日度聚合数据: 俄方 {ru_total} 件, 乌方 {ua_total} 件")

observed_ratio = ru_total / ua_total
n_days = 1559  # Oryx 双方数据天数
print(f"观测率比 ρ_obs = RU/UA = {observed_ratio:.3f}")

# 1b. 加载乌克兰官方声称数据（用于约束观测概率下界）
print(f"\n--- 利用声称数据约束观测概率下界 ---")
try:
    ua_claims = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'))
    # 坦克
    ws_tank_daily = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0)
    ws_tank_lam = ws_tank_daily['Tanks'].mean() if 'Tanks' in ws_tank_daily.columns else 0
    ua_tank_lam = ua_claims['Tanks'].mean() if 'Tanks' in ua_claims.columns else 0
    if ws_tank_lam > 0 and ua_tank_lam > 0:
        p_ru_tank_lower = ws_tank_lam / ua_tank_lam  # 验证/声称 = 观测概率下界
        print(f"  坦克: 验证 λ̂={ws_tank_lam:.2f}/天, 声称 λ̂={ua_tank_lam:.2f}/天")
        print(f"       观测概率下界 p_RU ≥ {p_ru_tank_lower:.3f} (≈{p_ru_tank_lower*100:.0f}%)")

    # 装甲车辆 (IFV + APC 简化为 APCs)
    ws_ifv = ws_tank_daily['Infantry fighting vehicles'].mean() if 'Infantry fighting vehicles' in ws_tank_daily.columns else 0
    ua_apc_lam = ua_claims['Armoured Personnel Carriers'].mean() if 'Armoured Personnel Carriers' in ua_claims.columns else 0
    ws_armor_lam = ws_tank_lam + ws_ifv  # 简化: 坦克+IFV vs 官方APC
    if ws_armor_lam > 0 and ua_apc_lam > 0:
        p_ru_armor_lower = ws_armor_lam / ua_apc_lam
        print(f"  装甲车辆: 验证 λ̂≈{ws_armor_lam:.2f}/天, 声称 λ̂={ua_apc_lam:.2f}/天")
        print(f"           观测概率下界 p_RU ≥ {p_ru_armor_lower:.3f} (≈{p_ru_armor_lower*100:.0f}%)")

    # 火炮
    ws_art = 0
    for col in ['Self-propelled artillery', 'Towed artillery']:
        if col in ws_tank_daily.columns:
            ws_art += ws_tank_daily[col].mean()
    ua_art_lam = ua_claims['Field Artillery'].mean() if 'Field Artillery' in ua_claims.columns else 0
    if ws_art > 0 and ua_art_lam > 0:
        p_ru_art_lower = ws_art / ua_art_lam
        print(f"  火炮:   验证 λ̂={ws_art:.2f}/天, 声称 λ̂={ua_art_lam:.2f}/天")
        print(f"          观测概率下界 p_RU ≥ {p_ru_art_lower:.3f} (≈{p_ru_art_lower*100:.0f}%)")

    print(f"\n  解读: 如果假设官方声称不会低估敌方损失（弱假设），")
    print(f"        则验证数/声称数 ≤ 观测概率 ≤ 1")
    print(f"        这为 p_RU 提供了数据驱动的下界约束")
except Exception as e:
    print(f"  声称数据加载失败: {e}")
    p_ru_tank_lower = 0.33  # fallback value
    p_ru_armor_lower = 0.35
    p_ru_art_lower = 0.03

# ===========================
# 2. 偏误校正网格构建
# ===========================
print(f"\n{'='*70}")
print("2. 偏误校正网格 (观测概率 → 修正率比)")
print("=" * 70)

# 使用更细的网格
obs_probs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for p_ru in obs_probs:
    for p_ua in obs_probs:
        corrected_ratio = (ru_total / p_ru) / (ua_total / p_ua)
        results.append({
            'p_RU': p_ru,
            'p_UA': p_ua,
            'observed_RU': ru_total,
            'observed_UA': ua_total,
            'corrected_RU': round(ru_total / p_ru, 1),
            'corrected_UA': round(ua_total / p_ua, 1),
            'observed_ratio': round(observed_ratio, 3),
            'corrected_ratio': round(corrected_ratio, 3),
            'ratio_still_above_1': corrected_ratio > 1.0,
            # 标记是否在合理范围内（考虑声称下界约束）
            'within_claims_bound': p_ru >= 0.3,  # 基于坦克声称/验证 ≈ 0.33
        })

sensitivity_df = pd.DataFrame(results)

# ===========================
# 3. 偏误校正后的率比推断
# ===========================
print(f"\n--- 3.1 偏误校正公式 ---")
print(f"  真实率比 ρ_true = ρ_obs × (p_UA / p_RU)")
print(f"  观测率比 ρ_obs = {observed_ratio:.3f}")
print(f"  逆转条件: p_UA / p_RU < {1/observed_ratio:.3f}")
print(f"  即: 当乌方观测概率不到俄方的 {1/observed_ratio*100:.1f}% 时，结论逆转")

print(f"\n--- 3.2 合理假设下的偏误校正区间 ---")
# 定义几个合理的观测概率假设范围
scenarios = [
    ("保守 (p_RU高, p_UA低)", 0.9, 0.7, 0.4, 0.5),
    ("中等 (双方接近)",      0.7, 0.8, 0.5, 0.7),
    ("乐观 (p_RU低, p_UA高)", 0.5, 0.7, 0.6, 0.8),
]
for label, ru_lo, ru_hi, ua_lo, ua_hi in scenarios:
    # 在该范围内，ρ_true 的下界和上界
    rho_lo = observed_ratio * (ua_lo / ru_hi)
    rho_hi = observed_ratio * (ua_hi / ru_lo)
    above_1 = "✓ 仍>1" if rho_lo > 1 else ("⚠ 可能≤1" if rho_hi > 1 else "✗ 已逆转")
    print(f"  {label}: p_RU∈[{ru_lo},{ru_hi}], p_UA∈[{ua_lo},{ua_hi}]")
    print(f"           ρ_true ∈ [{rho_lo:.2f}, {rho_hi:.2f}] {above_1}")

print(f"\n--- 3.3 关键场景（含声称下界约束 p_RU ≥ 0.33）---")
for p_ru in [0.6, 0.7, 0.8, 0.9]:
    for p_ua in [0.4, 0.5, 0.6]:
        cr = observed_ratio * (p_ua / p_ru)
        feasible = "✓" if p_ru >= 0.3 else "边界"
        marker = " ← 逆转!" if cr < 1.0 else ""
        print(f"  p_RU={p_ru:.1f}, p_UA={p_ua:.1f}: ρ_corrected={cr:.3f}{marker} [{feasible}]")

# ===========================
# 4. 基于 Bootstrap 的偏误校正 ★ NEW
# ===========================
print(f"\n{'='*70}")
print("4. 基于 Bootstrap 的偏误校正率比推断")
print("=" * 70)
print("方法: 对双方日度损失数据做 Block Bootstrap, 同时在每次重抽样中")
print("      从合理先验分布中抽取 p_RU 和 p_UA, 获得偏误校正后率比的")
print("      完整 Bootstrap 分布")

# 加载日度数据
daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
ru_daily_vals = daily['Russia_Total'].values
ua_daily_vals = daily['Ukraine_Total'].values
n = len(ru_daily_vals)

# Block Bootstrap 参数
B = 5000
SEED = 20260612
block_len = 11  # 与日度 Bootstrap 一致
random.seed(SEED)
np.random.seed(SEED)

# 观测概率的先验设定（基于声称数据约束和领域知识）
# p_RU: Beta 分布, 均值 ~0.7（装备损失大概率被记录）
#       利用坦克声称下界 0.33 约束，取 α/(α+β)=0.7, 且 p>0.33 的概率高
# p_UA: Beta 分布, 均值 ~0.55（乌方在俄占区损失更难被拍到）
#       取 α/(α+β)=0.55
# 使用 Beta(α, β) 分布，其中 α 控制均值，α+β 控制集中度

# 先验参数: Beta(α, β), 均值 = α/(α+β)
ru_prior_alpha, ru_prior_beta = 7, 3     # 均值 0.7, 中等集中
ua_prior_alpha, ua_prior_beta = 5.5, 4.5  # 均值 0.55, 中等集中

print(f"\n观测概率先验设定:")
print(f"  p_RU ~ Beta({ru_prior_alpha}, {ru_prior_beta}), 均值={ru_prior_alpha/(ru_prior_alpha+ru_prior_beta):.1f}")
print(f"  p_UA ~ Beta({ua_prior_alpha}, {ua_prior_beta}), 均值={ua_prior_alpha/(ua_prior_alpha+ua_prior_beta):.2f}")
print(f"  注: 先验均值基于以下考虑——")
print(f"      - 俄方损失多在乌控区(易拍摄)且多为大型装备 → p_RU 偏高")
print(f"      - 乌方损失部分在俄占区(难拍摄) → p_UA 偏低")
print(f"      - 声称数据给出 p_RU 不低于 ~0.33 (坦克)")

# Block Bootstrap + 偏误校正
k_blocks = int(np.ceil(n / block_len))
boot_rho_raw = np.zeros(B)      # 未校正的率比
boot_rho_corrected = np.zeros(B)  # 偏误校正后的率比

for b in range(B):
    # Block Bootstrap 重抽样俄方日度数据
    bs_ru = np.zeros(n)
    pos = 0
    for _ in range(k_blocks):
        start = np.random.randint(0, n - block_len + 1)
        block = ru_daily_vals[start:start + block_len]
        end = min(pos + len(block), n)
        bs_ru[pos:end] = block[:end - pos]
        pos = end
        if pos >= n:
            break
    lam_ru_boot = np.mean(bs_ru)

    # Block Bootstrap 重抽样乌方日度数据
    bs_ua = np.zeros(n)
    pos = 0
    for _ in range(k_blocks):
        start = np.random.randint(0, n - block_len + 1)
        block = ua_daily_vals[start:start + block_len]
        end = min(pos + len(block), n)
        bs_ua[pos:end] = block[:end - pos]
        pos = end
        if pos >= n:
            break
    lam_ua_boot = np.mean(bs_ua)

    boot_rho_raw[b] = lam_ru_boot / lam_ua_boot

    # 从先验中抽取观测概率
    p_ru_draw = np.random.beta(ru_prior_alpha, ru_prior_beta)
    p_ua_draw = np.random.beta(ua_prior_alpha, ua_prior_beta)

    # 偏误校正后的率比
    boot_rho_corrected[b] = boot_rho_raw[b] * (p_ua_draw / p_ru_draw)

# 结果统计
rho_obs_point = np.mean(ru_daily_vals) / np.mean(ua_daily_vals)
print(f"\n--- 4.1 偏误校正 Bootstrap 结果 ---")
print(f"  观测率比 ρ_obs (点估计):    {rho_obs_point:.3f}")
print(f"  Bootstrap 均值 (未校正):    {np.mean(boot_rho_raw):.3f}")
print(f"  Bootstrap 均值 (偏误校正):  {np.mean(boot_rho_corrected):.3f}")
print(f"  Bootstrap SE (未校正):      {np.std(boot_rho_raw, ddof=1):.4f}")
print(f"  Bootstrap SE (偏误校正):    {np.std(boot_rho_corrected, ddof=1):.4f}")

# 偏误校正后的置信区间
corr_ci_low = np.percentile(boot_rho_corrected, 2.5)
corr_ci_med = np.percentile(boot_rho_corrected, 50)
corr_ci_high = np.percentile(boot_rho_corrected, 97.5)
print(f"\n  偏误校正后率比 95% Bootstrap CI:")
print(f"    [{corr_ci_low:.3f}, {corr_ci_high:.3f}] (中位数: {corr_ci_med:.3f})")

# 核心指标: P(ρ_corrected > 1)
prob_above_1 = np.mean(boot_rho_corrected > 1)
print(f"\n  ★ P(ρ_corrected > 1 | 合理先验) = {prob_above_1:.1%}")
print(f"    解读: 即使在考虑观测偏误校正后，仍有 {prob_above_1:.0%} 的")
print(f"          Bootstrap 样本支持俄方真实损失率高于乌方")

# 分位点摘要
for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    val = np.percentile(boot_rho_corrected, q * 100)
    print(f"    {q:.0%} 分位点: {val:.3f}")

# ===========================
# 5. 分装备类型的偏误校正
# ===========================
print(f"\n{'='*70}")
print("5. 分装备类型偏误校正")
print("=" * 70)

try:
    cat_stats = df_oryx.groupby(['side', 'category'])['item_count'].sum().unstack(fill_value=0)
    ru_by_cat = cat_stats.loc['Russia'] if 'Russia' in cat_stats.index else None
    ua_by_cat = cat_stats.loc['Ukraine'] if 'Ukraine' in cat_stats.index else None

    if ru_by_cat is not None and ua_by_cat is not None:
        common_cats = ru_by_cat.index.intersection(ua_by_cat.index)
        print(f"  {'装备类型':<35s} {'RU观测':>6s} {'UA观测':>6s} {'观测率比':>8s} "
              f"{'逆转需p_UA/p_RU<':>14s} {'稳健性':>8s}")
        print(f"  {'-'*90}")

        for cat in sorted(common_cats, key=lambda c: ru_by_cat[c] / max(ua_by_cat[c], 1), reverse=True):
            ru_c = int(ru_by_cat[cat])
            ua_c = int(ua_by_cat[cat])
            if ru_c + ua_c < 20:
                continue
            cat_ratio = ru_c / max(ua_c, 1)
            # 逆转所需的 p_UA/p_RU 比值
            reversal_threshold = 1 / cat_ratio if cat_ratio > 1 else 999
            # 稳健性判断
            if reversal_threshold < 0.4:
                robustness = "极高"
            elif reversal_threshold < 0.6:
                robustness = "高"
            elif reversal_threshold < 0.8:
                robustness = "中等"
            elif reversal_threshold < 1.0:
                robustness = "较低"
            else:
                robustness = "已逆转*"
            print(f"  {cat:<35s} {ru_c:6d} {ua_c:6d} {cat_ratio:8.2f} "
                  f"{'< '+str(round(reversal_threshold,3)) if reversal_threshold < 900 else 'N/A':>14s} {robustness:>8s}")
except Exception as e:
    print(f"  分类型分析出错: {e}")

# ===========================
# 6. 偏误校正结论
# ===========================
print(f"\n{'='*70}")
print("偏误校正分析核心结论")
print("=" * 70)
print(f"""
  1. 观测率比 ρ_obs = {observed_ratio:.2f}（基于影像验证数据）

  2. 偏误校正公式: ρ_true = ρ_obs × (p_UA / p_RU)
     - p_RU: 俄方装备被影像验证的概率
     - p_UA: 乌方装备被影像验证的概率
     - 逆转条件: p_UA / p_RU < {1/observed_ratio:.3f}

  3. 声称数据约束: 俄方坦克观测概率 p_RU ≥ {p_ru_tank_lower:.2f}
     （基于"官方声称不会低估敌方损失"的弱假设）

  4. 合理先验下 (p_RU~Beta(7,3), p_UA~Beta(5.5,4.5)):
     - 偏误校正后率比中位数: {corr_ci_med:.2f}
     - 偏误校正后 95% CI: [{corr_ci_low:.2f}, {corr_ci_high:.2f}]
     - P(ρ_corrected > 1) = {prob_above_1:.0%}

  5. 结论稳健性: 在合理的观测偏误先验下，俄方真实装备损失率
     高于乌方的概率为 {prob_above_1:.0%}。仅当乌方观测概率系统性
     低于俄方 51.5% 时（即 p_UA/p_RU < 0.485），结论才可能逆转。
     该极端场景与俄占区地理现实存在一定张力。
""")

# ===========================
# 7. 保存结果
# ===========================
print(f"\n--- 保存结果 ---")
os.makedirs(os.path.join('output', 'tables'), exist_ok=True)

# 偏误校正网格
sensitivity_df.to_csv(os.path.join('output', 'tables', 'observation_sensitivity_overall.csv'), index=False)
print("✓ observation_sensitivity_overall.csv (偏误校正网格)")

# Bootstrap 偏误校正分布
bias_corrected_df = pd.DataFrame({
    'replicate': range(1, B + 1),
    'rho_raw': np.round(boot_rho_raw, 6),
    'rho_bias_corrected': np.round(boot_rho_corrected, 6),
})
bias_corrected_df.to_csv(os.path.join('output', 'tables', 'bias_corrected_ratio_bootstrap.csv'), index=False)
print(f"✓ bias_corrected_ratio_bootstrap.csv (B={B} Bootstrap 偏误校正分布)")

# 偏误校正汇总
summary = pd.DataFrame([{
    'observed_ratio': round(observed_ratio, 4),
    'ru_total': ru_total,
    'ua_total': ua_total,
    'n_days': n_days,
    'p_ru_prior_mean': round(ru_prior_alpha / (ru_prior_alpha + ru_prior_beta), 3),
    'p_ua_prior_mean': round(ua_prior_alpha / (ua_prior_alpha + ua_prior_beta), 3),
    'bootstrap_reps': B,
    'bias_corrected_median': round(corr_ci_med, 4),
    'bias_corrected_ci_low': round(corr_ci_low, 4),
    'bias_corrected_ci_high': round(corr_ci_high, 4),
    'prob_rho_above_1': round(prob_above_1, 4),
    'claims_bound_p_ru_tank': round(p_ru_tank_lower, 4) if 'p_ru_tank_lower' in dir() else None,
}])
summary.to_csv(os.path.join('output', 'tables', 'bias_correction_summary.csv'), index=False)
print("✓ bias_correction_summary.csv")

print(f"\n观测概率偏误校正分析全部完成!")
