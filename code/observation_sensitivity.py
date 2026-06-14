import pandas as pd
import numpy as np
from scipy import stats
import os, sys

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("Observation Probability Bias Correction & Sensitivity Analysis")
print("=" * 70)

try:
    df_oryx = pd.read_csv(os.path.join('data', 'oryx_losses_combined.csv'))
    ru_total = int(df_oryx[df_oryx['side'] == 'Russia']['item_count'].sum())
    ua_total = int(df_oryx[df_oryx['side'] == 'Ukraine']['item_count'].sum())
    print(f"Oryx item-level (item_count): RU {ru_total}, UA {ua_total}")
    try:
        daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
        ru_daily = int(daily['Russia_Total'].sum())
        ua_daily = int(daily['Ukraine_Total'].sum())
        if ru_daily != ru_total or ua_daily != ua_total:
            print(f"  Daily aggregated: RU {ru_daily}, UA {ua_daily}")
            print(f"  Diff: RU {ru_daily-ru_total:+d}, UA {ua_daily-ua_total:+d}")
    except:
        pass
except:
    daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
    ru_total = int(daily['Russia_Total'].sum())
    ua_total = int(daily['Ukraine_Total'].sum())
    print(f"Daily aggregated: RU {ru_total}, UA {ua_total}")

observed_ratio = ru_total / ua_total
n_days = 1559
print(f"Observed ratio: {observed_ratio:.3f}")

print(f"\n--- Using Claims to Bound Observation Probability ---")
p_ru_tank_lower = 0.33
p_ru_armor_lower = 0.35
p_ru_art_lower = 0.03
try:
    ua_claims = pd.read_csv(os.path.join('data', 'daily_ua_claims.csv'))
    ws_tank_daily = pd.read_csv(os.path.join('data', 'daily_by_type_ws.csv'), index_col=0)
    ws_tank_lam = ws_tank_daily['Tanks'].mean() if 'Tanks' in ws_tank_daily.columns else 0
    ua_tank_lam = ua_claims['Tanks'].mean() if 'Tanks' in ua_claims.columns else 0
    if ws_tank_lam > 0 and ua_tank_lam > 0:
        p_ru_tank_lower = ws_tank_lam / ua_tank_lam
        print(f"  Tanks: verified lambda={ws_tank_lam:.2f}/day, claimed lambda={ua_tank_lam:.2f}/day")
        print(f"         p_RU >= {p_ru_tank_lower:.3f} (~{p_ru_tank_lower*100:.0f}%)")

    ws_ifv = ws_tank_daily['Infantry fighting vehicles'].mean() if 'Infantry fighting vehicles' in ws_tank_daily.columns else 0
    ua_apc_lam = ua_claims['Armoured Personnel Carriers'].mean() if 'Armoured Personnel Carriers' in ua_claims.columns else 0
    ws_armor_lam = ws_tank_lam + ws_ifv
    if ws_armor_lam > 0 and ua_apc_lam > 0:
        p_ru_armor_lower = ws_armor_lam / ua_apc_lam
        print(f"  Armour: verified lambda~{ws_armor_lam:.2f}/day, claimed lambda={ua_apc_lam:.2f}/day")
        print(f"          p_RU >= {p_ru_armor_lower:.3f} (~{p_ru_armor_lower*100:.0f}%)")

    ws_art = 0
    for col in ['Self-propelled artillery', 'Towed artillery']:
        if col in ws_tank_daily.columns:
            ws_art += ws_tank_daily[col].mean()
    ua_art_lam = ua_claims['Field Artillery'].mean() if 'Field Artillery' in ua_claims.columns else 0
    if ws_art > 0 and ua_art_lam > 0:
        p_ru_art_lower = ws_art / ua_art_lam
        print(f"  Artillery: verified lambda={ws_art:.2f}/day, claimed lambda={ua_art_lam:.2f}/day")
        print(f"             p_RU >= {p_ru_art_lower:.3f} (~{p_ru_art_lower*100:.0f}%)")
except Exception as e:
    print(f"  Claims data load failed: {e}")

print(f"\n{'='*70}")
print("Bias Correction Grid (observation probability -> corrected ratio)")
print("=" * 70)

obs_probs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []
for p_ru in obs_probs:
    for p_ua in obs_probs:
        corrected_ratio = (ru_total / p_ru) / (ua_total / p_ua)
        results.append({
            'p_RU': p_ru, 'p_UA': p_ua,
            'observed_RU': ru_total, 'observed_UA': ua_total,
            'corrected_RU': round(ru_total / p_ru, 1),
            'corrected_UA': round(ua_total / p_ua, 1),
            'observed_ratio': round(observed_ratio, 3),
            'corrected_ratio': round(corrected_ratio, 3),
            'ratio_still_above_1': corrected_ratio > 1.0,
            'within_claims_bound': p_ru >= 0.3,
        })

sensitivity_df = pd.DataFrame(results)

print(f"\n--- Bias Correction Formula ---")
print(f"  True ratio = observed_ratio * (p_UA / p_RU)")
print(f"  Observed ratio = {observed_ratio:.3f}")
print(f"  Reversal condition: p_UA / p_RU < {1/observed_ratio:.3f}")

print(f"\n--- Corrected Ratio Under Reasonable Assumptions ---")
scenarios = [
    ("Conservative (p_RU high, p_UA low)", 0.9, 0.7, 0.4, 0.5),
    ("Moderate (similar)", 0.7, 0.8, 0.5, 0.7),
    ("Optimistic (p_RU low, p_UA high)", 0.5, 0.7, 0.6, 0.8),
]
for label, ru_lo, ru_hi, ua_lo, ua_hi in scenarios:
    rho_lo = observed_ratio * (ua_lo / ru_hi)
    rho_hi = observed_ratio * (ua_hi / ru_lo)
    above_1 = "still >1" if rho_lo > 1 else ("may <=1" if rho_hi > 1 else "reversed")
    print(f"  {label}: p_RU in [{ru_lo},{ru_hi}], p_UA in [{ua_lo},{ua_hi}]")
    print(f"           true ratio in [{rho_lo:.2f}, {rho_hi:.2f}] {above_1}")

print(f"\n--- Key Scenarios (with claims bound p_RU >= 0.33) ---")
for p_ru in [0.6, 0.7, 0.8, 0.9]:
    for p_ua in [0.4, 0.5, 0.6]:
        cr = observed_ratio * (p_ua / p_ru)
        feasible = "feasible" if p_ru >= 0.3 else "marginal"
        marker = " <-- REVERSAL!" if cr < 1.0 else ""
        print(f"  p_RU={p_ru:.1f}, p_UA={p_ua:.1f}: corrected ratio={cr:.3f}{marker} [{feasible}]")

print(f"\n{'='*70}")
print("Bootstrap-Based Bias-Corrected Ratio Inference")
print("=" * 70)
print("Method: Block bootstrap daily loss data for both sides, simultaneously")
print("draw p_RU and p_UA from reasonable priors, obtain full distribution of")
print("bias-corrected ratio.")

daily = pd.read_csv(os.path.join('data', 'daily_oryx_both_sides.csv'))
ru_daily_vals = daily['Russia_Total'].values
ua_daily_vals = daily['Ukraine_Total'].values
n = len(ru_daily_vals)

B = 5000
SEED = 20260612
block_len = 11

np.random.seed(SEED)

ru_prior_alpha, ru_prior_beta = 7, 3
ua_prior_alpha, ua_prior_beta = 5.5, 4.5

print(f"\nObservation probability priors:")
print(f"  p_RU ~ Beta({ru_prior_alpha}, {ru_prior_beta}), mean={ru_prior_alpha/(ru_prior_alpha+ru_prior_beta):.1f}")
print(f"  p_UA ~ Beta({ua_prior_alpha}, {ua_prior_beta}), mean={ua_prior_alpha/(ua_prior_alpha+ua_prior_beta):.2f}")

k_blocks = int(np.ceil(n / block_len))
boot_rho_raw = np.zeros(B)
boot_rho_corrected = np.zeros(B)

for b in range(B):
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

    p_ru_draw = np.random.beta(ru_prior_alpha, ru_prior_beta)
    p_ua_draw = np.random.beta(ua_prior_alpha, ua_prior_beta)
    boot_rho_corrected[b] = boot_rho_raw[b] * (p_ua_draw / p_ru_draw)

rho_obs_point = np.mean(ru_daily_vals) / np.mean(ua_daily_vals)
print(f"\n--- Bias-Corrected Bootstrap Results ---")
print(f"  Observed ratio (point est):   {rho_obs_point:.3f}")
print(f"  Bootstrap mean (uncorrected): {np.mean(boot_rho_raw):.3f}")
print(f"  Bootstrap mean (corrected):   {np.mean(boot_rho_corrected):.3f}")
print(f"  Bootstrap SE (uncorrected):   {np.std(boot_rho_raw, ddof=1):.4f}")
print(f"  Bootstrap SE (corrected):     {np.std(boot_rho_corrected, ddof=1):.4f}")

corr_ci_low = np.percentile(boot_rho_corrected, 2.5)
corr_ci_med = np.percentile(boot_rho_corrected, 50)
corr_ci_high = np.percentile(boot_rho_corrected, 97.5)
print(f"\n  Bias-corrected ratio 95% Bootstrap CI:")
print(f"    [{corr_ci_low:.3f}, {corr_ci_high:.3f}] (median: {corr_ci_med:.3f})")

prob_above_1 = np.mean(boot_rho_corrected > 1)
print(f"\n  P(corrected_ratio > 1 | reasonable prior) = {prob_above_1:.1%}")

for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    val = np.percentile(boot_rho_corrected, q * 100)
    print(f"    {q:.0%} quantile: {val:.3f}")

print(f"\n{'='*70}")
print("Bias Correction by Equipment Type")
print("=" * 70)

try:
    cat_stats = df_oryx.groupby(['side', 'category'])['item_count'].sum().unstack(fill_value=0)
    ru_by_cat = cat_stats.loc['Russia'] if 'Russia' in cat_stats.index else None
    ua_by_cat = cat_stats.loc['Ukraine'] if 'Ukraine' in cat_stats.index else None

    if ru_by_cat is not None and ua_by_cat is not None:
        common_cats = ru_by_cat.index.intersection(ua_by_cat.index)
        print(f"  {'Type':<35s} {'RU_obs':>6s} {'UA_obs':>6s} {'Ratio':>8s} "
              f"{'Reversal_need_pUA/pRU<':>18s} {'Robustness':>10s}")
        print(f"  {'-'*90}")

        for cat in sorted(common_cats, key=lambda c: ru_by_cat[c] / max(ua_by_cat[c], 1), reverse=True):
            ru_c = int(ru_by_cat[cat])
            ua_c = int(ua_by_cat[cat])
            if ru_c + ua_c < 20:
                continue
            cat_ratio = ru_c / max(ua_c, 1)
            reversal_threshold = 1 / cat_ratio if cat_ratio > 1 else 999
            if reversal_threshold < 0.4:
                robustness = "Very High"
            elif reversal_threshold < 0.6:
                robustness = "High"
            elif reversal_threshold < 0.8:
                robustness = "Medium"
            elif reversal_threshold < 1.0:
                robustness = "Low"
            else:
                robustness = "Reversed*"
            print(f"  {cat:<35s} {ru_c:6d} {ua_c:6d} {cat_ratio:8.2f} "
                  f"{'< '+str(round(reversal_threshold,3)) if reversal_threshold < 900 else 'N/A':>18s} {robustness:>10s}")
except Exception as e:
    print(f"  Type analysis failed: {e}")

print(f"\n{'='*70}")
print("Bias Correction Core Conclusions")
print("=" * 70)
print(f"""
  1. Observed ratio = {observed_ratio:.2f} (image-verified data)

  2. Correction formula: true_ratio = observed_ratio * (p_UA / p_RU)
     - p_RU: probability RU equipment is image-verified
     - p_UA: probability UA equipment is image-verified
     - Reversal condition: p_UA / p_RU < {1/observed_ratio:.3f}

  3. Claims bound: RU tank observation probability p_RU >= {p_ru_tank_lower:.2f}
     (under weak assumption that official claims don't undercount enemy losses)

  4. Under reasonable priors (p_RU~Beta(7,3), p_UA~Beta(5.5,4.5)):
     - Bias-corrected ratio median: {corr_ci_med:.2f}
     - Bias-corrected 95% CI: [{corr_ci_low:.2f}, {corr_ci_high:.2f}]
     - P(corrected_ratio > 1) = {prob_above_1:.0%}

  5. Conclusion robustness: Under reasonable observation bias priors,
     probability that true RU loss rate exceeds UA is {prob_above_1:.0%}.
     Conclusion would only reverse if UA observation probability is
     systematically below {1/observed_ratio*100:.1f}% of RU observation probability,
     a scenario in tension with geographic realities of the war.
""")

print(f"\n--- Saving Results ---")
os.makedirs(os.path.join('output', 'tables'), exist_ok=True)

sensitivity_df.to_csv(os.path.join('output', 'tables', 'observation_sensitivity_overall.csv'), index=False)
print("  observation_sensitivity_overall.csv")

bias_corrected_df = pd.DataFrame({
    'replicate': range(1, B + 1),
    'rho_raw': np.round(boot_rho_raw, 6),
    'rho_bias_corrected': np.round(boot_rho_corrected, 6),
})
bias_corrected_df.to_csv(os.path.join('output', 'tables', 'bias_corrected_ratio_bootstrap.csv'), index=False)
print(f"  bias_corrected_ratio_bootstrap.csv (B={B})")

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
    'claims_bound_p_ru_tank': round(p_ru_tank_lower, 4),
}])
summary.to_csv(os.path.join('output', 'tables', 'bias_correction_summary.csv'), index=False)
print("  bias_correction_summary.csv")

print(f"\nObservation probability bias correction analysis complete!")
