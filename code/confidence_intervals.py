import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from utils import wald_ci, score_ci, exact_ci, all_cis, KEY_PHASES, MAJOR_EQUIPMENT_TYPES

print("=" * 70)
print("Poisson Confidence Interval Analysis")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)

print("\n--- Overall CI ---")
all_cis(daily_ws['total_losses'].values, label="WarSpotting overall (all periods)")

print("\n--- Key Phase CI Comparison ---")
for phase, (start, end) in KEY_PHASES.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    all_cis(daily_ws[mask]['total_losses'].values, label=phase)

print("\n--- Major Equipment Type CI ---")
for eq_type in MAJOR_EQUIPMENT_TYPES:
    if eq_type in daily_by_type.columns:
        all_cis(daily_by_type[eq_type].values, label=eq_type)

print("\n--- Coverage Monte Carlo Simulation ---")
print("Setup: Poisson(lambda), n=30 days, N_sim=10000")

np.random.seed(42)

def simulate_coverage(true_lambda, n_days=30, n_sim=10000, alpha=0.05):
    wald_cover = score_cover = exact_cover = 0
    for _ in range(n_sim):
        data = np.random.poisson(true_lambda, n_days)
        w_l, w_u, _, _ = wald_ci(data, alpha)
        if w_l <= true_lambda <= w_u:
            wald_cover += 1
        s_l, s_u, _, _ = score_ci(data, alpha)
        if s_l <= true_lambda <= s_u:
            score_cover += 1
        e_l, e_u = exact_ci(data, alpha)
        if e_l <= true_lambda <= e_u:
            exact_cover += 1
    return wald_cover/n_sim, score_cover/n_sim, exact_cover/n_sim

lambda_values = [0.5, 1, 2, 5, 10, 15, 20, 50]
sim_results = []
for lam in lambda_values:
    w, s, e = simulate_coverage(lam, n_days=30, n_sim=10000)
    sim_results.append({'lambda': lam, 'Wald': w, 'Score': s, 'Exact': e})
    print(f"  lambda={lam:5.1f}: Wald={w:.3f}, Score={s:.3f}, Exact={e:.3f}")

sim_df = pd.DataFrame(sim_results)
sim_df.to_csv(os.path.join('output', 'tables', 'ci_coverage_simulation.csv'), index=False)
print("  coverage simulation saved")

print("\n--- Effect of Sample Size on CI Width ---")
true_lambda = 15.0
n_values = [7, 14, 30, 60, 90, 180, 365]
print(f"lambda_true = {true_lambda}")
for n in n_values:
    data = np.random.poisson(true_lambda, n)
    w_l, w_u, _, _ = wald_ci(data)
    e_l, e_u = exact_ci(data)
    print(f"  n={n:4d}: Wald width={w_u-w_l:.3f}, Exact width={e_u-e_l:.3f}")

print("\n--- Saving CI Comparison Data ---")
ci_comparison = []
for name, data in [('Overall', daily_ws['total_losses'].values)] + \
                   [(t, daily_by_type[t].values) for t in MAJOR_EQUIPMENT_TYPES if t in daily_by_type.columns]:
    w_l, w_u, lhat, se = wald_ci(data)
    s_l, s_u, _, _ = score_ci(data)
    e_l, e_u = exact_ci(data)
    ci_comparison.append({
        'Category': name, 'MLE': lhat,
        'Wald_Lower': w_l, 'Wald_Upper': w_u,
        'Score_Lower': s_l, 'Score_Upper': s_u,
        'Exact_Lower': e_l, 'Exact_Upper': e_u, 'SE': se
    })

ci_df = pd.DataFrame(ci_comparison)
ci_df.to_csv(os.path.join('output', 'tables', 'ci_comparison.csv'), index=False)
print("  output/tables/ci_comparison.csv")
print("\nCI analysis complete.")
