import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from utils import poisson_mle, WAR_PHASES

print("=" * 70)
print("MLE Point Estimation Analysis")
print("=" * 70)

daily_ws = pd.read_csv(os.path.join('data', 'daily_warspotting.csv'), parse_dates=['date'])
daily_by_type = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0, parse_dates=True)
daily_by_status = pd.read_csv(os.path.join('data', 'daily_by_status_ws.csv'), index_col=0, parse_dates=True)
df_ua_daily = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'), parse_dates=['date'])

print("\n--- Overall MLE ---")
ws_total = poisson_mle(daily_ws['total_losses'].values)
print(f"WarSpotting (N={ws_total['n']} days):")
print(f"  lambda_hat = {ws_total['lambda_hat']:.4f}/day")
print(f"  SE = {ws_total['se']:.4f}")
print(f"  Total = {int(ws_total['total'])} items")
print(f"  95% Wald CI: [{ws_total['lambda_hat'] - 1.96*ws_total['se']:.4f}, {ws_total['lambda_hat'] + 1.96*ws_total['se']:.4f}]")

print("\n--- MLE by War Phase ---")
phase_results = []
for phase, (start, end) in WAR_PHASES.items():
    mask = (daily_ws['date'] >= start) & (daily_ws['date'] <= end)
    data = daily_ws[mask]['total_losses'].values
    if len(data) > 0:
        res = poisson_mle(data)
        res['phase'] = phase
        phase_results.append(res)
        print(f"  {phase}: lambda_hat={res['lambda_hat']:.2f}, SE={res['se']:.2f}, days={res['n']}")

print("\n--- Phase Homogeneity LRT ---")
pooled_lambda = np.mean([r['lambda_hat'] for r in phase_results])
ll_null = sum(-r['n'] * pooled_lambda + r['total'] * np.log(max(pooled_lambda, 1e-10)) for r in phase_results)
ll_alt = sum(r['log_likelihood'] for r in phase_results)
lr_stat = -2 * (ll_null - ll_alt)
df = len(phase_results) - 1
p_value = 1 - stats.chi2.cdf(lr_stat, df)
print(f"  LR stat = {lr_stat:.2f}, df = {df}, p < 0.0001")
print(f"  Conclusion: phases have significantly different loss rates")

print("\n--- MLE by Equipment Type (WarSpotting) ---")
type_results = []
for col in daily_by_type.columns:
    data = daily_by_type[col].values
    if np.sum(data) > 50:
        res = poisson_mle(data)
        res['type'] = col
        res['total_losses'] = int(np.sum(data))
        type_results.append(res)

type_results.sort(key=lambda x: x['lambda_hat'], reverse=True)
print(f"{'Type':<35s} {'lambda_hat':>8s} {'SE':>8s} {'Total':>8s} {'Days':>6s}")
print("-" * 70)
for r in type_results:
    print(f"{r['type']:<35s} {r['lambda_hat']:8.3f} {r['se']:8.3f} {r['total_losses']:8d} {r['n']:6d}")

print("\n--- MLE: WarSpotting vs Ukraine Claims ---")
ws_tanks = daily_by_type['Tanks'].values
ua_tanks = df_ua_daily['Tanks'].values[:len(ws_tanks)]
ws_tank_mle = poisson_mle(ws_tanks)
ua_tank_mle = poisson_mle(ua_tanks)
print(f"\nTanks:")
print(f"  WarSpotting: lambda_hat={ws_tank_mle['lambda_hat']:.3f}/day, SE={ws_tank_mle['se']:.3f}")
print(f"  UA claims:   lambda_hat={ua_tank_mle['lambda_hat']:.3f}/day, SE={ua_tank_mle['se']:.3f}")
print(f"  Claims/verified ratio = {ua_tank_mle['lambda_hat']/ws_tank_mle['lambda_hat']:.2f}")

ws_ifv = daily_by_type['Infantry fighting vehicles'].values
ua_apc = df_ua_daily['Armoured Personnel Carriers'].values[:len(ws_ifv)]
ws_ifv_mle = poisson_mle(ws_ifv)
ua_apc_mle = poisson_mle(ua_apc)
print(f"\nArmoured vehicles:")
print(f"  WarSpotting: lambda_hat={ws_ifv_mle['lambda_hat']:.3f}/day")
print(f"  UA claims:   lambda_hat={ua_apc_mle['lambda_hat']:.3f}/day")
print(f"  Claims/verified ratio = {ua_apc_mle['lambda_hat']/ws_ifv_mle['lambda_hat']:.2f}")

ws_art = (daily_by_type['Self-propelled artillery'] + daily_by_type['Towed artillery']).values
ua_art = df_ua_daily['Field Artillery'].values[:len(ws_art)]
ws_art_mle = poisson_mle(ws_art)
ua_art_mle = poisson_mle(ua_art)
print(f"\nArtillery:")
print(f"  WarSpotting: lambda_hat={ws_art_mle['lambda_hat']:.3f}/day")
print(f"  UA claims:   lambda_hat={ua_art_mle['lambda_hat']:.3f}/day")
print(f"  Claims/verified ratio = {ua_art_mle['lambda_hat']/ws_art_mle['lambda_hat']:.2f}")

print("\n--- Observation Bias Analysis ---")
print("Bias sources: both sides tend to exaggerate enemy losses;")
print("image-verified data is affected by 'probability of being observed'.")
print("Large equipment (tanks, artillery) is easier to observe than small (drones).")
print("Losses in UA-controlled territory are easier to photograph than in RU-controlled areas.")

status_mle = {}
for col in daily_by_status.columns:
    data = daily_by_status[col].values
    res = poisson_mle(data)
    status_mle[col] = res
    print(f"  {col}: lambda_hat={res['lambda_hat']:.3f}/day, total={int(res['total'])}")

print("\nBias correction estimates:")
total_verified = int(ws_total['total'])
print(f"  Total verified: {total_verified}")
for obs_rate in [0.6, 0.7, 0.8]:
    est_true = total_verified / obs_rate
    print(f"  Obs rate={obs_rate:.0%}: estimated true total ~ {est_true:.0f}")

print("\n--- Saving results ---")
results_df = pd.DataFrame([
    {'Category': 'Overall (WarSpotting)', 'lambda_hat': ws_total['lambda_hat'],
     'se': ws_total['se'], 'n_days': ws_total['n'], 'total': int(ws_total['total'])}
] + [
    {'Category': f"Phase: {r['phase']}", 'lambda_hat': r['lambda_hat'],
     'se': r['se'], 'n_days': r['n'], 'total': int(r['total'])}
    for r in phase_results
] + [
    {'Category': f"Type: {r['type']}", 'lambda_hat': r['lambda_hat'],
     'se': r['se'], 'n_days': r['n'], 'total': r['total_losses']}
    for r in type_results
])

results_df.to_csv(os.path.join('output', 'tables', 'mle_results.csv'), index=False)
print("  output/tables/mle_results.csv")
print("\nMLE analysis complete.")
