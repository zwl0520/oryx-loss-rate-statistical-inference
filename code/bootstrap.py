import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

from utils import choose_block_length, block_bootstrap, bca_ci, BOOTSTRAP_PHASES

print("=" * 70)
print("Block Bootstrap Analysis")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_ua = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'), parse_dates=['date'])

print("\n--- Block Length Selection ---")
ws_data = daily_ws['total_losses'].values
n = len(ws_data)
b_opt = choose_block_length(ws_data)
print(f"Data length: n = {n}")
print(f"Recommended block length: b = {b_opt}")

x = ws_data - np.mean(ws_data)
var = np.sum(x**2)
acf_vals = [np.sum(x[lag:] * x[:-lag]) / var for lag in range(1, 11)]
print(f"Autocorrelation (lag 1-10): {[f'{a:.3f}' for a in acf_vals]}")

print("\n--- Overall Bootstrap CI ---")
theta_hat = np.mean(ws_data)
print(f"MLE lambda_hat = {theta_hat:.4f}")

print("Running IID Bootstrap (B=2000)...")
iid_boot = np.array([np.mean(np.random.choice(ws_data, size=n, replace=True)) for _ in range(2000)])
iid_pct_ci = (np.percentile(iid_boot, 2.5), np.percentile(iid_boot, 97.5))
print(f"  IID Bootstrap 95% CI: [{iid_pct_ci[0]:.4f}, {iid_pct_ci[1]:.4f}]")

print(f"Running Block Bootstrap (b={b_opt}, B=2000)...")
bca_result = bca_ci(ws_data, B=2000, block_len=b_opt)
print(f"  Block Bootstrap mean: {bca_result['boot_mean']:.4f}")
print(f"  Block Bootstrap SE: {bca_result['boot_se']:.4f}")
print(f"  Bias correction z0 = {bca_result['z0']:.4f}")
print(f"  Acceleration a_hat = {bca_result['a_hat']:.4f}")
print(f"  Percentile 95% CI: [{bca_result['percentile_ci'][0]:.4f}, {bca_result['percentile_ci'][1]:.4f}]")
print(f"  BCa 95% CI: [{bca_result['lower']:.4f}, {bca_result['upper']:.4f}]")

print("\n--- Bootstrap by Phase ---")
phase_bs_results = []
for phase, (start, end) in BOOTSTRAP_PHASES.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    data_p = daily_ws[mask]['total_losses'].values
    if len(data_p) < 20:
        continue
    b = choose_block_length(data_p, max_lag=15)
    res = bca_ci(data_p, B=2000, block_len=b)
    phase_bs_results.append({
        'Phase': phase, 'MLE': np.mean(data_p),
        'BCa_Lower': res['lower'], 'BCa_Upper': res['upper'],
        'Bootstrap_SE': res['boot_se'], 'z0': res['z0'], 'a_hat': res['a_hat'],
    })
    print(f"  {phase}: MLE={np.mean(data_p):.2f}, BCa 95% CI=[{res['lower']:.2f}, {res['upper']:.2f}], SE={res['boot_se']:.2f}")

print("\n--- IID vs Block Bootstrap SE by Block Length ---")
block_lengths = [1, 3, 5, 7, 10, 14, 20, 30]
for bl in block_lengths:
    if bl <= 1:
        boot_se = np.std(iid_boot, ddof=1)
    else:
        bm = block_bootstrap(ws_data, B=2000, block_len=bl)
        boot_se = np.std(bm, ddof=1)
    print(f"  b={bl:2d}: Bootstrap SE = {boot_se:.4f}")

print("\n--- Bootstrap Two-Phase Difference Test ---")
p1_mask = (daily_ws['date'] >= '2022-02-24') & (daily_ws['date'] <= '2022-04-30')
p9_mask = (daily_ws['date'] >= '2025-01-01') & (daily_ws['date'] <= '2026-06-03')
p1_data = daily_ws[p1_mask]['total_losses'].values
p9_data = daily_ws[p9_mask]['total_losses'].values
obs_diff = np.mean(p1_data) - np.mean(p9_data)
print(f"Phase 1 mean: {np.mean(p1_data):.2f}")
print(f"Phase 9 mean: {np.mean(p9_data):.2f}")
print(f"Observed difference: {obs_diff:.2f}")

B = 2000
diff_boot = np.zeros(B)
b1 = choose_block_length(p1_data, max_lag=15)
b9 = choose_block_length(p9_data, max_lag=15)
for i in range(B):
    bs1 = block_bootstrap(p1_data, B=1, block_len=b1)[0]
    bs9 = block_bootstrap(p9_data, B=1, block_len=b9)[0]
    diff_boot[i] = bs1 - bs9

diff_ci = np.percentile(diff_boot, [2.5, 97.5])
p_val = 2 * min(np.mean(diff_boot <= 0), np.mean(diff_boot >= 0))
print(f"Bootstrap 95% CI for difference: [{diff_ci[0]:.2f}, {diff_ci[1]:.2f}]")
print(f"Bootstrap p-value: {p_val:.4f}")

print("\n--- Saving Bootstrap Results ---")
bs_summary = pd.DataFrame(phase_bs_results)
bs_summary.to_csv(os.path.join('output', 'tables', 'bootstrap_results.csv'), index=False)
boot_dist_df = pd.DataFrame({
    'IID_Bootstrap_Means': iid_boot,
    'Block_Bootstrap_Means': bca_result['boot_means']
})
boot_dist_df.to_csv(os.path.join('output', 'tables', 'bootstrap_distributions.csv'), index=False)
print("  bootstrap_results.csv, bootstrap_distributions.csv")
print("\nBlock Bootstrap analysis complete.")
