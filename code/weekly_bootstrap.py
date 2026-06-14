import pandas as pd
import numpy as np
from scipy import stats
import os, sys

sys.stdout.reconfigure(encoding='utf-8')

from utils import choose_block_length, MAJOR_EQUIPMENT_TYPES

print("=" * 70)
print("Weekly Block Bootstrap Analysis (B=10000)")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_ws['week'] = (daily_ws['date'].dt.isocalendar().year.astype(str) + '-W' +
                    daily_ws['date'].dt.isocalendar().week.astype(str).str.zfill(2))

weekly = daily_ws.groupby('week').agg(
    week_start=('date', 'min'),
    loss_count=('total_losses', 'sum'),
).reset_index()
weekly = weekly.sort_values('week_start').reset_index(drop=True)

n_weeks = len(weekly)
counts = weekly['loss_count'].values
print(f"Weeks: {n_weeks}")
print(f"Weekly mean: {np.mean(counts):.1f}/week")
print(f"Daily mean (weekly basis): {np.sum(counts) / (7 * n_weeks):.3f}/day")

B = 10000
SEED = 20260605
ALPHA = 0.05
np.random.seed(SEED)

theta_hat = np.sum(counts) / (7 * n_weeks)
print(f"\ntheta_hat (MLE, weekly basis) = {theta_hat:.4f}/day")

b_weekly = choose_block_length(counts, max_lag=12)
print(f"Weekly block length: b = {b_weekly} weeks")

x_w = counts - np.mean(counts)
var_w = np.sum(x_w**2)
acf_w = [round(np.sum(x_w[lag:] * x_w[:-lag]) / var_w, 3)
         for lag in range(1, min(9, n_weeks // 4))]
print(f"Weekly ACF (lag 1-{len(acf_w)}): {acf_w}")

boot_rates = np.zeros(B)
k_blocks = int(np.ceil(n_weeks / b_weekly))
for b in range(B):
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
print(f"Bootstrap SE (weekly blocks): {boot_se:.4f}")
print(f"Percentile 95% CI: [{boot_percentile_ci[0]:.4f}, {boot_percentile_ci[1]:.4f}]")

print(f"\nComparison:")
try:
    bs_results = pd.read_csv(os.path.join('output', 'tables', 'bootstrap_results.csv'))
    daily_se = bs_results['Bootstrap_SE'].iloc[0] if 'Bootstrap_SE' in bs_results.columns else None
except:
    daily_se = None

if daily_se is not None:
    print(f"  Daily MBB SE: {daily_se:.4f} (WarSpotting, B=2000)")
    print(f"  Weekly MBB SE: {boot_se:.4f} (WarSpotting, b={b_weekly}w, B={B})")
    print(f"  Daily IID Bootstrap SE: 0.4168")
    print(f"  Weekly MBB / Daily IID ratio: {boot_se/0.4168:.2f}x")
else:
    print(f"  Weekly MBB SE: {boot_se:.4f} (WarSpotting, b={b_weekly}w, B={B})")

print(f"\n--- Jackknife + BCa ---")
jack_stats = np.zeros(n_weeks)
total_sum = np.sum(counts)
for i in range(n_weeks):
    jack_stats[i] = (total_sum - counts[i]) / (7 * (n_weeks - 1))

jack_mean = np.mean(jack_stats)

prop_less = np.clip(np.mean(boot_rates < theta_hat), 1/(2*B), 1 - 1/(2*B))
z0 = stats.norm.ppf(prop_less)

diffs = jack_mean - jack_stats
numerator = np.sum(diffs ** 3)
denominator = 6 * (np.sum(diffs ** 2) ** 1.5)
a_hat = numerator / denominator if denominator > 1e-10 else 0.0

def bca_quantile(prob):
    z = stats.norm.ppf(prob)
    denom = 1 - a_hat * (z0 + z)
    if denom <= 0:
        return 0.0 if prob < 0.5 else 1.0
    return np.clip(stats.norm.cdf(z0 + (z0 + z) / denom), 0.0, 1.0)

q_low = bca_quantile(ALPHA / 2)
q_high = bca_quantile(1 - ALPHA / 2)

bca_low = np.percentile(boot_rates, q_low * 100)
bca_high = np.percentile(boot_rates, q_high * 100)

print(f"Bias correction z0: {z0:.4f}")
print(f"Acceleration a: {a_hat:.4f}")
print(f"BCa adjusted quantiles: [{q_low:.4f}, {q_high:.4f}]")
print(f"BCa 95% CI: [{bca_low:.4f}, {bca_high:.4f}]")

print(f"\n--- Weekly Bootstrap by Equipment Type ---")
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_by_type['week'] = (daily_by_type.index.isocalendar().year.astype(str) + '-W' +
                          daily_by_type.index.isocalendar().week.astype(str).str.zfill(2))

type_boot_results = []
for eq_type in MAJOR_EQUIPMENT_TYPES:
    if eq_type not in daily_by_type.columns:
        continue
    weekly_type = daily_by_type.groupby('week')[eq_type].sum()
    type_counts = weekly_type.values

    if np.sum(type_counts) < 50:
        continue

    theta_t = np.sum(type_counts) / (7 * len(type_counts))
    n_t = len(type_counts)

    b_t = max(choose_block_length(type_counts, max_lag=8), 1)
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
    print(f"  {eq_type:<30s}: lambda_hat={theta_t:.3f}/day, SE={np.std(bt, ddof=1):.3f}")

print(f"\n--- Saving Weekly Bootstrap Results ---")
os.makedirs(os.path.join('output', 'tables'), exist_ok=True)

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

dist_df = pd.DataFrame({
    'replicate': range(1, B + 1),
    'rate_per_day': np.round(boot_rates, 6),
    'rate_per_30_days': np.round(boot_rates * 30, 6),
})
dist_df.to_csv(os.path.join('output', 'tables', 'weekly_bootstrap_distribution.csv'), index=False)

jk_df = pd.DataFrame({
    'omitted_week': weekly['week'].values,
    'omitted_week_start': weekly['week_start'].values,
    'omitted_loss_count': counts,
    'jackknife_rate_per_day': np.round(jack_stats, 6),
})
jk_df.to_csv(os.path.join('output', 'tables', 'weekly_jackknife_rates.csv'), index=False)

type_df = pd.DataFrame(type_boot_results)
type_df.to_csv(os.path.join('output', 'tables', 'weekly_bootstrap_by_type.csv'), index=False)

print("  weekly_bootstrap_bca.csv, weekly_bootstrap_distribution.csv")
print("  weekly_jackknife_rates.csv, weekly_bootstrap_by_type.csv")
print(f"\nWeekly Block Bootstrap complete!")
print(f"\nCore result comparison:")
print(f"  Weekly MBB BCa CI: [{bca_low:.4f}, {bca_high:.4f}] (WarSpotting, b={b_weekly}w, B={B})")
