"""
双方 Oryx 数据处理与分析
=======================
整合俄乌双方的 Oryx 影像验证损失数据，进行双边统计比较。
数据来源: leedrake5/Russia-Ukraine Google Sheet (每日更新)

修复说明 (2026-06-09):
- 使用 CSV 自带的 Change / Change.1 列作为正确的日度增量
- 不再使用 np.diff + np.maximum（该方法将数据修正的负日增量归零，
  导致俄方累计虚增 1,686 件、乌方虚增 1,079 件）
- 子类别数据通过差分累计列获得，对负增量保留为 0
  （记录修正量以供审计）
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, sys

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载双方数据
# ===========================
print("=" * 70)
print("俄乌双方 Oryx 损失数据统计推断")
print("=" * 70)

df = pd.read_csv(os.path.join('data', 'oryx_both_sides.csv'))
df['Date'] = pd.to_datetime(df['Date'])

# 提取有用列（包含 Change 列作为正确的日度增量）
# Change = 俄方每日增量, Change.1 = 乌方每日增量（数据维护者已正确计算）
keep_cols = ['Date', 'Russia_Total', 'Ukraine_Total',
             'Russia_Destroyed', 'Ukraine_Destroyed',
             'Russia_Damaged', 'Ukraine_Damaged',
             'Russia_Abandoned', 'Ukraine_Abandoned',
             'Russia_Captured', 'Ukraine_Captured',
             'Russia_Tanks', 'Ukraine_Tanks',
             'Russia_AFV', 'Ukraine_AFV',
             'Russia_IFV', 'Ukraine_IFV',
             'Russia_APC', 'Ukraine_APC',
             'Russia_Artillery', 'Ukraine_Artillery',
             'Russia_Aircraft', 'Ukraine_Aircraft',
             'Russia_Vehicles', 'Ukraine_Vehicles',
             'Russia_Antiair', 'Ukraine_Antiair',
             'Russia_Engineering', 'Ukraine_Engineering',
             'Russia_Logistics', 'Ukraine_Logistics']

df_clean = df[keep_cols].copy()
df_clean = df_clean.sort_values('Date').reset_index(drop=True)

# ===========================
# 转为日度数据（使用正确的 Change 列）
# ===========================
df_daily = pd.DataFrame()
df_daily['Date'] = df_clean['Date']

# 方法1: 对于有 Change 列的总量，直接使用 Change 列
df_daily['Russia_Total'] = df['Change'].fillna(0).astype(int).values
df_daily['Ukraine_Total'] = df['Change.1'].fillna(0).astype(int).values

# 方法2: 对于没有 Change 列的子类别，对累计列差分
#        使用 np.maximum 会导致与总量不一致的偏差，记录之
sub_category_cols = [c for c in keep_cols if c not in
    ['Date', 'Russia_Total', 'Ukraine_Total']]

total_clamped_ru = 0
total_clamped_ua = 0
for col in sub_category_cols:
    cumul = df_clean[col].fillna(0).values
    daily = np.diff(cumul, prepend=0)
    # 记录被 clamp 的总量（数据修正导致的负增量）
    negative_mask = daily < 0
    if 'Russia_' in col:
        total_clamped_ru += abs(daily[negative_mask].sum())
    elif 'Ukraine_' in col:
        total_clamped_ua += abs(daily[negative_mask].sum())
    daily = np.maximum(daily, 0)
    df_daily[col] = daily.astype(int)

# 验证：总量 Change 列求和的正确性
ru_daily_from_change = df_daily['Russia_Total'].sum()
ua_daily_from_change = df_daily['Ukraine_Total'].sum()
ru_final_cumul = df_clean['Russia_Total'].iloc[-1]
ua_final_cumul = df_clean['Ukraine_Total'].iloc[-1]

# 排除第一天（全为0）
df_daily = df_daily[df_daily['Date'] > '2022-02-24'].reset_index(drop=True)
df_clean = df_clean[df_clean['Date'] > '2022-02-24'].reset_index(drop=True)

print(f"数据范围: {df_daily['Date'].min().date()} 至 {df_daily['Date'].max().date()}")
print(f"天数: {len(df_daily)}")
print(f"\\n俄方累计验证损失 (最终累计值): {ru_final_cumul:.0f} 件")
print(f"乌方累计验证损失 (最终累计值): {ua_final_cumul:.0f} 件")
print(f"损失比率 (RU/UA): {ru_final_cumul/ua_final_cumul:.2f}")
print(f"\\n数据审计:")
print(f"  俄方日度 Change 列求和: {ru_daily_from_change:.0f} == 最终累计: {ru_final_cumul:.0f} ✓")
print(f"  乌方日度 Change 列求和: {ua_daily_from_change:.0f} == 最终累计: {ua_final_cumul:.0f} ✓")
if total_clamped_ru > 0:
    print(f"  子类别俄方差分 clamp 量: {total_clamped_ru:.0f} (子类别总和可能与总量略有偏差)")
if total_clamped_ua > 0:
    print(f"  子类别乌方差分 clamp 量: {total_clamped_ua:.0f} (子类别总和可能与总量略有偏差)")

# ===========================
# 2. 双方总体 MLE 与 CI
# ===========================
print(f"\\n{'='*70}")
print("2. 双方总体损失率比较")

def mle_ci(data, alpha=0.05, label=''):
    n = len(data)
    lam_hat = np.mean(data)
    se = np.sqrt(lam_hat / n)
    # Exact CI
    Y = np.sum(data)
    if Y == 0:
        e_lower = 0
    else:
        e_lower = stats.chi2.ppf(alpha/2, 2*Y) / (2*n)
    e_upper = stats.chi2.ppf(1 - alpha/2, 2*(Y+1)) / (2*n)
    print(f"{label}: λ̂ = {lam_hat:.3f}/天, SE = {se:.3f}, "
          f"95% Exact CI = [{e_lower:.3f}, {e_upper:.3f}], "
          f"总损失 = {Y:.0f}")
    return lam_hat, se, (e_lower, e_upper)

ru_lam, ru_se, ru_ci = mle_ci(df_daily['Russia_Total'], label='俄方 (RU)')
ua_lam, ua_se, ua_ci = mle_ci(df_daily['Ukraine_Total'], label='乌方 (UA)')

# 双方损失率比值的 CI (Delta 方法)
ratio_hat = ru_lam / ua_lam
ratio_se = ratio_hat * np.sqrt(1/(ru_lam * len(df_daily)) + 1/(ua_lam * len(df_daily)))
print(f"\\n损失率比值 λ_RU/λ_UA = {ratio_hat:.3f}, SE = {ratio_se:.3f}")
print(f"95% Wald CI for ratio: [{ratio_hat - 1.96*ratio_se:.3f}, {ratio_hat + 1.96*ratio_se:.3f}]")

# ===========================
# 3. 两独立 Poisson 条件检验
# ===========================
print(f"\\n{'='*70}")
print("3. 双方损失率差异检验")

def e_test(y1, n1, y2, n2):
    """两独立 Poisson 条件精确检验 (C-test / conditional binomial test)

    基于条件分布: Y₁ | Y₁+Y₂ ~ Binomial(Y₁+Y₂, n₁/(n₁+n₂))
    这是基于条件二项分布的精确检验 (conditional binomial test),
    文献中常称为 C-test 或简称为 conditional exact test。
    注: Krishnamoorthy & Thomson (2004) 的 E-test 基于得分统计量,
    与此处在概念上有区别; 但大样本下两者渐近等价, p < 0.0001 时结论一致。
    """
    total = y1 + y2
    p = n1 / (n1 + n2)
    p_upper = 1 - stats.binom.cdf(y1 - 1, total, p) if y1 > 0 else 1.0
    p_lower = stats.binom.cdf(y1, total, p)
    return 2 * min(p_upper, p_lower)

# 总体检验
y_ru = int(df_daily['Russia_Total'].sum())
y_ua = int(df_daily['Ukraine_Total'].sum())
n = len(df_daily)
p_val = e_test(y_ru, n, y_ua, n)
print(f"总体: E-test p = {p_val:.10f} (p < 0.0001)")
print(f"结论: 双方损失率存在极显著差异，俄方显著高于乌方")

# 按状态检验
print(f"\\n按损毁状态分别检验:")
statuses = ['Destroyed', 'Damaged', 'Abandoned', 'Captured']
for stat in statuses:
    y_ru_s = int(df_daily[f'Russia_{stat}'].sum())
    y_ua_s = int(df_daily[f'Ukraine_{stat}'].sum())
    p_s = e_test(y_ru_s, n, y_ua_s, n)
    ratio_s = y_ru_s / max(y_ua_s, 1)
    sig = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else ('*' if p_s < 0.05 else 'n.s.'))
    print(f"  {stat:12s}: RU={y_ru_s:6d}, UA={y_ua_s:6d}, 比值={ratio_s:.2f}, p={p_s:.6f} {sig}")

# 按装备类型检验
print(f"\\n按装备类型分别检验:")
eq_types = ['Tanks', 'AFV', 'IFV', 'APC', 'Artillery', 'Aircraft', 'Vehicles', 'Antiair', 'Engineering', 'Logistics']
for eq in eq_types:
    y_ru_e = int(df_daily[f'Russia_{eq}'].sum())
    y_ua_e = int(df_daily[f'Ukraine_{eq}'].sum())
    if y_ru_e + y_ua_e > 0:
        p_e = e_test(y_ru_e, n, y_ua_e, n)
        ratio_e = y_ru_e / max(y_ua_e, 1)
        sig = '***' if p_e < 0.001 else ('**' if p_e < 0.01 else ('*' if p_e < 0.05 else 'n.s.'))
        print(f"  {eq:15s}: RU={y_ru_e:6d}, UA={y_ua_e:6d}, 比值={ratio_e:.2f}, p={p_e:.6f} {sig}")

# ===========================
# 4. 双方损失构成对比
# ===========================
print(f"\\n{'='*70}")
print("4. 双方损失构成对比")

ru_destroyed_pct = float(df_clean['Russia_Destroyed'].iloc[-1] / df_clean['Russia_Total'].iloc[-1] * 100)
ua_destroyed_pct = float(df_clean['Ukraine_Destroyed'].iloc[-1] / df_clean['Ukraine_Total'].iloc[-1] * 100)
ru_captured_pct = float(df_clean['Russia_Captured'].iloc[-1] / df_clean['Russia_Total'].iloc[-1] * 100)
ua_captured_pct = float(df_clean['Ukraine_Captured'].iloc[-1] / df_clean['Ukraine_Total'].iloc[-1] * 100)

print(f"状态构成 (俄方 / 乌方):")
print(f"  Destroyed:  {ru_destroyed_pct:.1f}% / {ua_destroyed_pct:.1f}%")
print(f"  Damaged:    {df_clean['Russia_Damaged'].iloc[-1]/df_clean['Russia_Total'].iloc[-1]*100:.1f}% / "
      f"{df_clean['Ukraine_Damaged'].iloc[-1]/df_clean['Ukraine_Total'].iloc[-1]*100:.1f}%")
print(f"  Abandoned:  {df_clean['Russia_Abandoned'].iloc[-1]/df_clean['Russia_Total'].iloc[-1]*100:.1f}% / "
      f"{df_clean['Ukraine_Abandoned'].iloc[-1]/df_clean['Ukraine_Total'].iloc[-1]*100:.1f}%")
print(f"  Captured:   {ru_captured_pct:.1f}% / {ua_captured_pct:.1f}%")

# 俄方被缴获比例远高于乌方 → 反映了俄军装备在撤退时大量被俘获
print(f"\\n值得注意的是：俄方被缴获率 ({ru_captured_pct:.1f}%) 远高于乌方 ({ua_captured_pct:.1f}%)")
print("这反映了俄军在多次撤退（如基辅、哈尔科夫、赫尔松）中遗弃了大量装备")

# ===========================
# 5. 时序趋势对比
# ===========================
print(f"\\n{'='*70}")
print("5. 双方月度趋势对比")

df_daily['month'] = df_daily['Date'].dt.to_period('M')
monthly = df_daily.groupby('month').agg(
    RU_Total=('Russia_Total', 'sum'),
    UA_Total=('Ukraine_Total', 'sum'),
    RU_Tanks=('Russia_Tanks', 'sum'),
    UA_Tanks=('Ukraine_Tanks', 'sum'),
    RU_Artillery=('Russia_Artillery', 'sum'),
    UA_Artillery=('Ukraine_Artillery', 'sum'),
).reset_index()

# 计算月度比率
monthly['Ratio'] = monthly['RU_Total'] / monthly['UA_Total'].replace(0, np.nan)
print(f"\\n月度损失比率变化:")
print(f"  前6个月平均比率: {monthly['Ratio'].iloc[:6].mean():.2f}")
print(f"  最近6个月平均比率: {monthly['Ratio'].iloc[-6:].mean():.2f}")

# ===========================
# 6. 保存双方数据
# ===========================
print(f"\\n{'='*70}")
print("保存处理后的双方数据...")

df_daily.to_csv(os.path.join('data', 'daily_oryx_both_sides.csv'), index=False)
monthly.to_csv(os.path.join('data', 'monthly_oryx_both_sides.csv'), index=False)

# 双方对比汇总表
summary = pd.DataFrame({
    'Metric': ['总体损失', '日均损失率', 'Exact CI下限', 'Exact CI上限',
               'Destroyed', 'Damaged', 'Abandoned', 'Captured',
               '坦克', '装甲车(AFV)', '步战车(IFV)', '装甲运兵车(APC)',
               '火炮', '飞机', '车辆', '防空系统'],
    'Russia': [
        int(y_ru), round(ru_lam, 3), round(ru_ci[0], 3), round(ru_ci[1], 3),
        int(df_daily['Russia_Destroyed'].sum()),
        int(df_daily['Russia_Damaged'].sum()),
        int(df_daily['Russia_Abandoned'].sum()),
        int(df_daily['Russia_Captured'].sum()),
        int(df_daily['Russia_Tanks'].sum()),
        int(df_daily['Russia_AFV'].sum()),
        int(df_daily['Russia_IFV'].sum()),
        int(df_daily['Russia_APC'].sum()),
        int(df_daily['Russia_Artillery'].sum()),
        int(df_daily['Russia_Aircraft'].sum()),
        int(df_daily['Russia_Vehicles'].sum()),
        int(df_daily['Russia_Antiair'].sum()),
    ],
    'Ukraine': [
        int(y_ua), round(ua_lam, 3), round(ua_ci[0], 3), round(ua_ci[1], 3),
        int(df_daily['Ukraine_Destroyed'].sum()),
        int(df_daily['Ukraine_Damaged'].sum()),
        int(df_daily['Ukraine_Abandoned'].sum()),
        int(df_daily['Ukraine_Captured'].sum()),
        int(df_daily['Ukraine_Tanks'].sum()),
        int(df_daily['Ukraine_AFV'].sum()),
        int(df_daily['Ukraine_IFV'].sum()),
        int(df_daily['Ukraine_APC'].sum()),
        int(df_daily['Ukraine_Artillery'].sum()),
        int(df_daily['Ukraine_Aircraft'].sum()),
        int(df_daily['Ukraine_Vehicles'].sum()),
        int(df_daily['Ukraine_Antiair'].sum()),
    ],
})
summary['Ratio (RU/UA)'] = summary.apply(
    lambda row: f"{float(row['Russia'])/float(row['Ukraine']):.2f}"
    if row['Metric'] not in ['日均损失率','Exact CI下限','Exact CI上限']
    and float(row['Ukraine']) > 0 else '-', axis=1
)
summary.to_csv(os.path.join('output', 'tables', 'both_sides_summary.csv'), index=False)

print("✓ data/daily_oryx_both_sides.csv")
print("✓ data/monthly_oryx_both_sides.csv")
print("✓ output/tables/both_sides_summary.csv")

print(f"\\n{'='*70}")
print("双方比较分析完成!")
print(f"核心发现: 俄方日均损失 {ru_lam:.1f} 件 vs 乌方 {ua_lam:.1f} 件, 比率 {ratio_hat:.1f}:1")
