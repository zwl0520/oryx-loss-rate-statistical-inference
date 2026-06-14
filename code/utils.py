import numpy as np
from scipy import stats

WAR_PHASES = {
    'Phase 1: Initial Invasion (2022.2-4)': ('2022-02-24', '2022-04-30'),
    'Phase 2: Donbas Offensive (2022.5-8)': ('2022-05-01', '2022-08-31'),
    'Phase 3: Counteroffensive (2022.9-12)': ('2022-09-01', '2022-12-31'),
    'Phase 4: Bakhmut/Winter (2023.1-5)': ('2023-01-01', '2023-05-31'),
    'Phase 5: Summer 2023 (2023.6-9)': ('2023-06-01', '2023-09-30'),
    'Phase 6: Avdiivka Campaign (2023.10-2024.2)': ('2023-10-01', '2024-02-29'),
    'Phase 7: Spring 2024 (2024.3-7)': ('2024-03-01', '2024-07-31'),
    'Phase 8: Kursk & Eastern Front (2024.8-12)': ('2024-08-01', '2024-12-31'),
    'Phase 9: Recent (2025.1-2026.6)': ('2025-01-01', '2026-06-30'),
}

KEY_PHASES = {
    'Phase 1: Initial Invasion': ('2022-02-24', '2022-04-30'),
    'Phase 3: Counteroffensive': ('2022-09-01', '2022-12-31'),
    'Phase 9: Recent': ('2025-01-01', '2026-06-30'),
}

BOOTSTRAP_PHASES = {
    'Phase 1 (Initial Invasion)': ('2022-02-24', '2022-04-30'),
    'Phase 3 (Counteroffensive)': ('2022-09-01', '2022-12-31'),
    'Phase 7 (Spring 2024)': ('2024-03-01', '2024-07-31'),
    'Phase 9 (Recent)': ('2025-01-01', '2026-06-30'),
}

TEST_PHASES = {
    'P1 (2022.2-4)': ('2022-02-24', '2022-04-30'),
    'P3 (2022.9-12)': ('2022-09-01', '2022-12-31'),
    'P7 (2024.3-7)': ('2024-03-01', '2024-07-31'),
    'P9 (2025-26)': ('2025-01-01', '2026-06-30'),
}

VIZ_PHASES = {
    'P1\n(2022.2-4)': ('2022-02-24', '2022-04-30'),
    'P2\n(2022.5-8)': ('2022-05-01', '2022-08-31'),
    'P3\n(2022.9-12)': ('2022-09-01', '2022-12-31'),
    'P4\n(2023.1-5)': ('2023-01-01', '2023-05-31'),
    'P5\n(2023.6-9)': ('2023-06-01', '2023-09-30'),
    'P6\n(2023.10-2024.2)': ('2023-10-01', '2024-02-29'),
    'P7\n(2024.3-7)': ('2024-03-01', '2024-07-31'),
    'P8\n(2024.8-12)': ('2024-08-01', '2024-12-31'),
    'P9\n(2025-26)': ('2025-01-01', '2026-06-30'),
}

MAJOR_EQUIPMENT_TYPES = [
    'Tanks', 'Infantry fighting vehicles', 'Self-propelled artillery',
    'Towed artillery', 'Drones', 'Transport'
]

STATUS_TYPES = ['Destroyed', 'Damaged', 'Abandoned', 'Captured']


def poisson_mle(data):
    n = len(data)
    lambda_hat = np.mean(data)
    se = np.sqrt(lambda_hat / n)
    total = np.sum(data)
    from scipy.special import gammaln
    log_lik = (-n * lambda_hat + total * np.log(max(lambda_hat, 1e-10))
               - np.sum(gammaln(data + 1)))
    return {
        'n': n, 'lambda_hat': lambda_hat, 'se': se,
        'total': total, 'var_lambda': lambda_hat / n,
        'log_likelihood': log_lik
    }


def wald_ci(data, alpha=0.05):
    n = len(data)
    lambda_hat = np.mean(data)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(lambda_hat / n)
    lower = max(0, lambda_hat - z * se)
    upper = lambda_hat + z * se
    return lower, upper, lambda_hat, se


def score_ci(data, alpha=0.05):
    n = len(data)
    lambda_hat = np.mean(data)
    z = stats.norm.ppf(1 - alpha / 2)
    bias = z**2 / (2 * n)
    center = lambda_hat + bias
    margin = z * np.sqrt(lambda_hat / n + z**2 / (4 * n**2))
    lower = max(0, center - margin)
    upper = center + margin
    return lower, upper, center, margin


def exact_ci(data, alpha=0.05):
    n = len(data)
    Y = np.sum(data)
    if Y == 0:
        lower = 0.0
    else:
        lower = stats.chi2.ppf(alpha / 2, 2 * int(Y)) / (2 * n)
    upper = stats.chi2.ppf(1 - alpha / 2, 2 * (int(Y) + 1)) / (2 * n)
    return lower, upper


def all_cis(data, alpha=0.05, label=""):
    w_l, w_u, lhat, se = wald_ci(data, alpha)
    s_l, s_u, _, _ = score_ci(data, alpha)
    e_l, e_u = exact_ci(data, alpha)
    print(f"\n{label}")
    print(f"  n={len(data)}, total={np.sum(data):.0f}")
    print(f"  MLE lambda_hat={lhat:.4f}, SE={se:.4f}")
    print(f"  Wald 95% CI:  [{w_l:.4f}, {w_u:.4f}]  width={w_u-w_l:.4f}")
    print(f"  Score 95% CI: [{s_l:.4f}, {s_u:.4f}]  width={s_u-s_l:.4f}")
    print(f"  Exact 95% CI: [{e_l:.4f}, {e_u:.4f}]  width={e_u-e_l:.4f}")
    return (w_l, w_u), (s_l, s_u), (e_l, e_u)


def choose_block_length(data, max_lag=30):
    n = len(data)
    x = data - np.mean(data)
    var = np.sum(x**2)
    if var < 1e-10:
        return 1
    for lag in range(1, min(max_lag, n // 4)):
        acf_val = np.sum(x[lag:] * x[:-lag]) / var
        if abs(acf_val) < 2 / np.sqrt(n):
            return max(lag, 2)
    return max(int(n ** (1/3)), 2)


def block_bootstrap(data, B=2000, block_len=None):
    n = len(data)
    if block_len is None:
        block_len = choose_block_length(data)
    k = int(np.ceil(n / block_len))
    boot_means = np.zeros(B)
    for i in range(B):
        bs = np.zeros(n)
        pos = 0
        for _ in range(k):
            start = np.random.randint(0, n - block_len + 1)
            block = data[start:start + block_len]
            end = min(pos + len(block), n)
            bs[pos:end] = block[:end - pos]
            pos = end
            if pos >= n:
                break
        boot_means[i] = np.mean(bs)
    return boot_means


def jackknife_means(data):
    n = len(data)
    total = np.sum(data)
    jk = np.zeros(n)
    for i in range(n):
        jk[i] = (total - data[i]) / (n - 1)
    return jk


def bca_ci(data, B=2000, alpha=0.05, block_len=None):
    n = len(data)
    theta_hat = np.mean(data)
    boot_means = block_bootstrap(data, B, block_len)
    z0 = stats.norm.ppf(np.clip(np.mean(boot_means < theta_hat), 1/(2*B), 1 - 1/(2*B)))
    jk_means = jackknife_means(data)
    jk_mean_overall = np.mean(jk_means)
    diffs = jk_mean_overall - jk_means
    numerator = np.sum(diffs ** 3)
    denominator = 6 * (np.sum(diffs ** 2) ** 1.5)
    a_hat = numerator / denominator if denominator > 1e-10 else 0.0
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a_hat * (z0 + z_1_alpha)))
    lower = np.percentile(boot_means, alpha1 * 100)
    upper = np.percentile(boot_means, alpha2 * 100)
    return {
        'lower': lower, 'upper': upper,
        'z0': z0, 'a_hat': a_hat,
        'boot_means': boot_means,
        'boot_mean': np.mean(boot_means),
        'boot_se': np.std(boot_means, ddof=1),
        'percentile_ci': (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)),
    }


def e_test(y1, n1, y2, n2):
    total = y1 + y2
    p = n1 / (n1 + n2)
    p_upper = 1 - stats.binom.cdf(y1 - 1, total, p) if y1 > 0 else 1.0
    p_lower = stats.binom.cdf(y1, total, p)
    return 2 * min(p_upper, p_lower)


def poisson_lrt(data1, data2):
    n1, n2 = len(data1), len(data2)
    lambda1 = np.mean(data1)
    lambda2 = np.mean(data2)
    lambda0 = (np.sum(data1) + np.sum(data2)) / (n1 + n2)
    ll1 = -n1*lambda1 + np.sum(data1)*np.log(max(lambda1, 1e-10))
    ll2 = -n2*lambda2 + np.sum(data2)*np.log(max(lambda2, 1e-10))
    ll_alt = ll1 + ll2
    total_data = np.concatenate([data1, data2])
    ll_null = -(n1+n2)*lambda0 + np.sum(total_data)*np.log(max(lambda0, 1e-10))
    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    return lr_stat, p_value


def poisson_conditional_test(data1, data2):
    n1, n2 = len(data1), len(data2)
    y1, y2 = int(np.sum(data1)), int(np.sum(data2))
    lambda1, lambda2 = np.mean(data1), np.mean(data2)
    e_pval = e_test(y1, n1, y2, n2)
    lr_stat, lr_pval = poisson_lrt(data1, data2)
    rate_ratio = lambda1 / lambda2 if lambda2 > 0 else np.inf
    se_diff = np.sqrt(lambda1/n1 + lambda2/n2)
    diff = lambda1 - lambda2
    return {
        'n1': n1, 'n2': n2,
        'total1': y1, 'total2': y2,
        'lambda1': lambda1, 'lambda2': lambda2,
        'rate_ratio': rate_ratio,
        'diff': diff,
        'diff_ci': (diff - 1.96 * se_diff, diff + 1.96 * se_diff),
        'e_test_pval': e_pval,
        'lrt_stat': lr_stat,
        'lrt_pval': lr_pval,
    }


def macro_region(lat, lon, short=False):
    if pd.isna(lat) or pd.isna(lon):
        return 'No Coordinates' if short else 'Missing Coordinates'
    if lon >= 36:
        return 'East' if short else 'East (卢甘斯克-顿涅茨克)'
    if lon < 28:
        return 'West' if short else 'West (远离前线)'
    if lat >= 50:
        return 'North' if short else 'North (哈尔科夫-苏梅-基辅)'
    if lat < 47.5:
        return 'South' if short else 'South (赫尔松-扎波罗热-克里米亚)'
    return 'Central' if short else 'Central (顿巴斯核心区)'


import pandas as pd


def assign_phase(date_val, phases=None):
    if phases is None:
        phases = WAR_PHASES
    for phase_name, (start, end) in phases.items():
        if pd.Timestamp(start) <= date_val <= pd.Timestamp(end):
            return phase_name
    return 'Other'


def mle_ci_summary(data, alpha=0.05, label=''):
    n = len(data)
    lam_hat = np.mean(data)
    se = np.sqrt(lam_hat / n)
    Y = np.sum(data)
    if Y == 0:
        e_lower = 0
    else:
        e_lower = stats.chi2.ppf(alpha/2, 2*int(Y)) / (2*n)
    e_upper = stats.chi2.ppf(1 - alpha/2, 2*(int(Y)+1)) / (2*n)
    print(f"{label}: lambda_hat={lam_hat:.3f}/day, SE={se:.3f}, "
          f"95% Exact CI=[{e_lower:.3f}, {e_upper:.3f}], total={Y:.0f}")
    return lam_hat, se, (e_lower, e_upper)
