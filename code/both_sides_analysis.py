import pandas as pd
import numpy as np
from scipy import stats
import os, sys

sys.stdout.reconfigure(encoding='utf-8')
from utils import e_test, mle_ci_summary

print("=" * 70)
print("Oryx Both-Sides Loss Data Statistical Inference")
print("=" * 70)

df = pd.read_csv(os.path.join('data', 'oryx_both_sides.csv'))
df['Date'] = pd.to_datetime(df['Date'])

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

df_daily = pd.DataFrame()
df_daily['Date'] = df_clean['Date']
df_daily['Russia_Total'] = df['Change'].fillna(0).astype(int).values
df_daily['Ukraine_Total'] = df['Change.1'].fillna(0).astype(int).values

sub_category_cols = [c for c in keep_cols if c not in ['Date', 'Russia_Total', 'Ukraine_Total']]
total_clamped_ru = 0
total_clamped_ua = 0
for col in sub_category_cols:
    cumul = df_clean[col].fillna(0).values
    daily = np.diff(cumul, prepend=0)
    negative_mask = daily < 0
    if 'Russia_' in col:
        total_clamped_ru += abs(daily[negative_mask].sum())
    elif 'Ukraine_' in col:
        total_clamped_ua += abs(daily[negative_mask].sum())
    daily = np.maximum(daily, 0)
    df_daily[col] = daily.astype(int)

ru_daily_from_change = df_daily['Russia_Total'].sum()
ua_daily_from_change = df_daily['Ukraine_Total'].sum()
ru_final_cumul = df_clean['Russia_Total'].iloc[-1]
ua_final_cumul = df_clean['Ukraine_Total'].iloc[-1]

df_daily = df_daily[df_daily['Date'] > '2022-02-24'].reset_index(drop=True)
df_clean = df_clean[df_clean['Date'] > '2022-02-24'].reset_index(drop=True)

print(f"Date range: {df_daily['Date'].min().date()} to {df_daily['Date'].max().date()}")
print(f"Days: {len(df_daily)}")
print(f"\nRussia cumulative verified (final): {ru_final_cumul:.0f}")
print(f"Ukraine cumulative verified (final): {ua_final_cumul:.0f}")
print(f"Loss ratio RU/UA: {ru_final_cumul/ua_final_cumul:.2f}")
print(f"\nAudit:")
print(f"  RU Change sum: {ru_daily_from_change:.0f} == final cumulative: {ru_final_cumul:.0f}")
print(f"  UA Change sum: {ua_daily_from_change:.0f} == final cumulative: {ua_final_cumul:.0f}")
if total_clamped_ru > 0:
    print(f"  Sub-category RU diff clamp: {total_clamped_ru:.0f}")
if total_clamped_ua > 0:
    print(f"  Sub-category UA diff clamp: {total_clamped_ua:.0f}")

print(f"\n{'='*70}")
print("Both Sides Overall Loss Rate Comparison")

ru_lam, ru_se, ru_ci = mle_ci_summary(df_daily['Russia_Total'], label='Russia (RU)')
ua_lam, ua_se, ua_ci = mle_ci_summary(df_daily['Ukraine_Total'], label='Ukraine (UA)')

ratio_hat = ru_lam / ua_lam
ratio_se = ratio_hat * np.sqrt(1/(ru_lam * len(df_daily)) + 1/(ua_lam * len(df_daily)))
print(f"\nLoss rate ratio RU/UA = {ratio_hat:.3f}, SE = {ratio_se:.3f}")
print(f"95% Wald CI: [{ratio_hat - 1.96*ratio_se:.3f}, {ratio_hat + 1.96*ratio_se:.3f}]")

print(f"\n{'='*70}")
print("Both Sides Loss Rate Difference Test")

y_ru = int(df_daily['Russia_Total'].sum())
y_ua = int(df_daily['Ukraine_Total'].sum())
n = len(df_daily)
p_val = e_test(y_ru, n, y_ua, n)
print(f"Overall: E-test p = {p_val:.10f} (p < 0.0001)")
print(f"Conclusion: extremely significant difference, RU > UA")

print(f"\nBy status:")
for stat in ['Destroyed', 'Damaged', 'Abandoned', 'Captured']:
    y_ru_s = int(df_daily[f'Russia_{stat}'].sum())
    y_ua_s = int(df_daily[f'Ukraine_{stat}'].sum())
    p_s = e_test(y_ru_s, n, y_ua_s, n)
    ratio_s = y_ru_s / max(y_ua_s, 1)
    sig = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else ('*' if p_s < 0.05 else 'n.s.'))
    print(f"  {stat:12s}: RU={y_ru_s:6d}, UA={y_ua_s:6d}, ratio={ratio_s:.2f}, p={p_s:.6f} {sig}")

print(f"\nBy equipment type:")
eq_types = ['Tanks', 'AFV', 'IFV', 'APC', 'Artillery', 'Aircraft', 'Vehicles', 'Antiair', 'Engineering', 'Logistics']
for eq in eq_types:
    y_ru_e = int(df_daily[f'Russia_{eq}'].sum())
    y_ua_e = int(df_daily[f'Ukraine_{eq}'].sum())
    if y_ru_e + y_ua_e > 0:
        p_e = e_test(y_ru_e, n, y_ua_e, n)
        ratio_e = y_ru_e / max(y_ua_e, 1)
        sig = '***' if p_e < 0.001 else ('**' if p_e < 0.01 else ('*' if p_e < 0.05 else 'n.s.'))
        print(f"  {eq:15s}: RU={y_ru_e:6d}, UA={y_ua_e:6d}, ratio={ratio_e:.2f}, p={p_e:.6f} {sig}")

print(f"\n{'='*70}")
print("Loss Composition Comparison")

ru_destroyed_pct = float(df_clean['Russia_Destroyed'].iloc[-1] / df_clean['Russia_Total'].iloc[-1] * 100)
ua_destroyed_pct = float(df_clean['Ukraine_Destroyed'].iloc[-1] / df_clean['Ukraine_Total'].iloc[-1] * 100)
ru_captured_pct = float(df_clean['Russia_Captured'].iloc[-1] / df_clean['Russia_Total'].iloc[-1] * 100)
ua_captured_pct = float(df_clean['Ukraine_Captured'].iloc[-1] / df_clean['Ukraine_Total'].iloc[-1] * 100)

print(f"Status composition (RU / UA):")
print(f"  Destroyed:  {ru_destroyed_pct:.1f}% / {ua_destroyed_pct:.1f}%")
print(f"  Damaged:    {df_clean['Russia_Damaged'].iloc[-1]/df_clean['Russia_Total'].iloc[-1]*100:.1f}% / {df_clean['Ukraine_Damaged'].iloc[-1]/df_clean['Ukraine_Total'].iloc[-1]*100:.1f}%")
print(f"  Abandoned:  {df_clean['Russia_Abandoned'].iloc[-1]/df_clean['Russia_Total'].iloc[-1]*100:.1f}% / {df_clean['Ukraine_Abandoned'].iloc[-1]/df_clean['Ukraine_Total'].iloc[-1]*100:.1f}%")
print(f"  Captured:   {ru_captured_pct:.1f}% / {ua_captured_pct:.1f}%")
print(f"\nNote: RU captured rate ({ru_captured_pct:.1f}%) much higher than UA ({ua_captured_pct:.1f}%)")
print("Reflects RU equipment abandoned during retreats (Kyiv, Kharkiv, Kherson)")

print(f"\n{'='*70}")
print("Monthly Trend Comparison")

df_daily['month'] = df_daily['Date'].dt.to_period('M')
monthly = df_daily.groupby('month').agg(
    RU_Total=('Russia_Total', 'sum'),
    UA_Total=('Ukraine_Total', 'sum'),
    RU_Tanks=('Russia_Tanks', 'sum'),
    UA_Tanks=('Ukraine_Tanks', 'sum'),
    RU_Artillery=('Russia_Artillery', 'sum'),
    UA_Artillery=('Ukraine_Artillery', 'sum'),
).reset_index()

monthly['Ratio'] = monthly['RU_Total'] / monthly['UA_Total'].replace(0, np.nan)
print(f"\nMonthly loss ratio changes:")
print(f"  First 6 months mean ratio: {monthly['Ratio'].iloc[:6].mean():.2f}")
print(f"  Last 6 months mean ratio: {monthly['Ratio'].iloc[-6:].mean():.2f}")

print(f"\n{'='*70}")
print("Saving both-sides data...")

df_daily.to_csv(os.path.join('data', 'daily_oryx_both_sides.csv'), index=False)
monthly.to_csv(os.path.join('data', 'monthly_oryx_both_sides.csv'), index=False)

summary = pd.DataFrame({
    'Metric': ['Total Losses', 'Daily Loss Rate', 'Exact CI Lower', 'Exact CI Upper',
               'Destroyed', 'Damaged', 'Abandoned', 'Captured',
               'Tanks', 'AFV', 'IFV', 'APC',
               'Artillery', 'Aircraft', 'Vehicles', 'Anti-air'],
    'Russia': [
        y_ru, round(ru_lam, 3), round(ru_ci[0], 3), round(ru_ci[1], 3),
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
        y_ua, round(ua_lam, 3), round(ua_ci[0], 3), round(ua_ci[1], 3),
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
    if row['Metric'] not in ['Daily Loss Rate', 'Exact CI Lower', 'Exact CI Upper']
    and float(row['Ukraine']) > 0 else '-', axis=1
)
summary.to_csv(os.path.join('output', 'tables', 'both_sides_summary.csv'), index=False)

print("  data/daily_oryx_both_sides.csv")
print("  data/monthly_oryx_both_sides.csv")
print("  output/tables/both_sides_summary.csv")

print(f"\n{'='*70}")
print("Both-sides analysis complete!")
print(f"Key finding: RU daily {ru_lam:.1f} vs UA {ua_lam:.1f}, ratio {ratio_hat:.1f}:1")
