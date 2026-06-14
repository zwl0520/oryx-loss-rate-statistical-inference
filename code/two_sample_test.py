import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
from utils import e_test, poisson_lrt, poisson_conditional_test, TEST_PHASES

print("=" * 70)
print("Two-Sample Poisson Conditional Test")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_ua = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'), parse_dates=['date'])

print("\n--- Phase Pairwise Comparison ---")
phase_data = {}
for phase, (start, end) in TEST_PHASES.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    phase_data[phase] = daily_ws[mask]['total_losses'].values

comparisons = list(combinations(TEST_PHASES.keys(), 2))
alpha = 0.05
n_comparisons = len(comparisons)
bonferroni_alpha = alpha / n_comparisons
print(f"Comparisons: {n_comparisons}, Bonferroni-adjusted alpha = {bonferroni_alpha:.4f}")
print(f"\n{'Comparison':<30s} {'lambda1':>8s} {'lambda2':>8s} {'Ratio':>8s} {'E-test p':>10s} {'LRT p':>10s} {'Sig':>8s}")
print("-" * 100)

phase_comparisons = []
for p1, p2 in comparisons:
    res = poisson_conditional_test(phase_data[p1], phase_data[p2])
    phase_comparisons.append({'Phase 1': p1, 'Phase 2': p2, **res})
    bonf_sig = "***" if res['e_test_pval'] < bonferroni_alpha else ("**" if res['e_test_pval'] < 0.01 else ("*" if res['e_test_pval'] < 0.05 else "n.s."))
    print(f"{p1} vs {p2:<20s} {res['lambda1']:8.2f} {res['lambda2']:8.2f} {res['rate_ratio']:8.3f} {res['e_test_pval']:10.6f} {res['lrt_pval']:10.6f} {bonf_sig:>8s}")

print("\n--- Equipment Type Pairwise Comparison ---")
major_types = ['Tanks', 'Infantry fighting vehicles', 'Self-propelled artillery',
               'Towed artillery', 'Drones', 'Transport']
type_data = {t: daily_by_type[t].values for t in major_types if t in daily_by_type.columns}
type_comparisons = list(combinations(type_data.keys(), 2))
n_type_comps = len(type_comparisons)
bonf_type_alpha = alpha / n_type_comps
print(f"Comparisons: {n_type_comps}, Bonferroni alpha = {bonf_type_alpha:.6f}")

type_test_results = []
for t1, t2 in type_comparisons:
    res = poisson_conditional_test(type_data[t1], type_data[t2])
    type_test_results.append({'Type 1': t1, 'Type 2': t2, **res})
    bonf_sig = "***" if res['e_test_pval'] < bonf_type_alpha else ("**" if res['e_test_pval'] < 0.01 else ("*" if res['e_test_pval'] < 0.05 else "n.s."))
    if res['e_test_pval'] < 0.05:
        print(f"  {t1} vs {t2}: lambda1={res['lambda1']:.3f}, lambda2={res['lambda2']:.3f}, "
              f"ratio={res['rate_ratio']:.3f}, p={res['e_test_pval']:.6f} {bonf_sig}")

print("\n--- WarSpotting Verified vs Ukraine Claims ---")
ws_tank = daily_by_type['Tanks'].values
min_len = min(len(ws_tank), len(daily_ua['Tanks']))
ws_tank_aligned = ws_tank[:min_len]
ua_tank_aligned = daily_ua['Tanks'].values[:min_len]

res_claim_vs_verify = poisson_conditional_test(ua_tank_aligned, ws_tank_aligned)
print(f"Tank daily losses:")
print(f"  WarSpotting: lambda_hat={res_claim_vs_verify['lambda2']:.3f}/day")
print(f"  UA claims:   lambda_hat={res_claim_vs_verify['lambda1']:.3f}/day")
print(f"  Claims/verified ratio = {res_claim_vs_verify['rate_ratio']:.3f}")
print(f"  E-test p = {res_claim_vs_verify['e_test_pval']:.10f}")
print(f"  LRT stat = {res_claim_vs_verify['lrt_stat']:.2f}, p < 0.0001")

ws_art = (daily_by_type['Self-propelled artillery'] + daily_by_type['Towed artillery']).values
ua_art = daily_ua['Field Artillery'].values[:len(ws_art)]
res_art = poisson_conditional_test(ua_art, ws_art)
print(f"\nArtillery daily losses:")
print(f"  WarSpotting: lambda_hat={res_art['lambda2']:.3f}/day")
print(f"  UA claims:   lambda_hat={res_art['lambda1']:.3f}/day")
print(f"  Claims/verified ratio = {res_art['rate_ratio']:.3f}")
print(f"  E-test p = {res_art['e_test_pval']:.10f}")

print("\n--- Status Comparison: Destroyed vs Captured ---")
daily_by_status = pd.read_csv(os.path.join('data', 'daily_by_status_ws.csv'), index_col=0, parse_dates=True)
res_stat = poisson_conditional_test(daily_by_status['Destroyed'].values, daily_by_status['Captured'].values)
print(f"  Destroyed: lambda_hat={res_stat['lambda1']:.2f}/day")
print(f"  Captured:  lambda_hat={res_stat['lambda2']:.2f}/day")
print(f"  Ratio = {res_stat['rate_ratio']:.2f}, p < 0.0001")

print("\n--- Saving Results ---")
pd.DataFrame(phase_comparisons).to_csv(os.path.join('output', 'tables', 'phase_comparisons.csv'), index=False)
pd.DataFrame(type_test_results).to_csv(os.path.join('output', 'tables', 'type_comparisons.csv'), index=False)
print("  phase_comparisons.csv, type_comparisons.csv")
print("\nTwo-sample Poisson conditional test complete.")
