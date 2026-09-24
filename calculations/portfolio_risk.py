"""Portfolio risk, sizing, and performance diagnostics as pure functions.

Formulas (one line each):
mean_variance_objective: w'mu - (risk_aversion / 2) w'Sw
unconstrained_mv_weights: inv(S) mu / risk_aversion
minimum_variance_objective: w'Sw
marginal_risk_contribution: Sw / sqrt(w'Sw)
component_risk_contribution: w * (Sw / sqrt(w'Sw))
portfolio_variance: w'Sw
systematic_variance: e'Fe with e = B'w
garch_update: omega + alpha shock^2 + beta prev_var
garch_variance: iterate garch_update over returns with zero-mean shocks
volatility_target_scale: min(target_vol / forecast_vol, cap)
gross_exposure: sum |w|
net_exposure: sum w
beta_adjusted_net: sum w beta
equal_weights: 1 / count per asset
cap_weights: iterative water-filling cap, budget preserved
score_weights: s / sum(s)
inverse_vol_weights: (1 / v) / sum(1 / v)
diversification_ratio: sum(w v) / sqrt(w'Sw)
tracking_error: std(port - bench) * annualization
active_weights: w_port - w_bench
portfolio_turnover: sum |new - old| / 2
kelly_binary: prob / loss - (1 - prob) / win
kelly_objective: prob log(1 + f win) + (1 - prob) log(1 - f loss)
sample_covariance: sum((a - ma)(b - mb)) / (n - ddof)
sample_correlation: cov / (std_a std_b)
ewma_cov_update: decay prev + (1 - decay) r r'
realized_volatility_hf: sqrt(sum r^2) * annualization
shrinkage_covariance: delta target + (1 - delta) sample
skewness: m3 / (m2 sqrt(m2))
excess_kurtosis: m4 / m2^2 - (1 + 2)
active_share: sum |w_port - w_bench| / 2
value_at_risk_historical: -(1 - confidence) quantile of returns
value_at_risk_parametric: z_score std - mean
expected_shortfall: -mean(returns at or below the VaR quantile); negative means gain
max_drawdown: max (peak - equity) / peak
time_under_water: longest run of periods below the running peak
stress_pnl: sum w shock
tail_risk_measure: mean(shortfall - cutoff)
marginal_var: z_score Sw / sqrt(w'Sw)
component_var: w * marginal_var

Sign convention: VaR, expected shortfall, and drawdown are reported as
positive losses. Long/short weights keep their signs everywhere else.
"""

from __future__ import annotations

import math

__all__ = [
    "mean_variance_objective",
    "unconstrained_mv_weights",
    "minimum_variance_objective",
    "marginal_risk_contribution",
    "component_risk_contribution",
    "portfolio_variance",
    "systematic_variance",
    "garch_update",
    "garch_variance",
    "volatility_target_scale",
    "gross_exposure",
    "net_exposure",
    "beta_adjusted_net",
    "equal_weights",
    "cap_weights",
    "score_weights",
    "inverse_vol_weights",
    "diversification_ratio",
    "tracking_error",
    "active_weights",
    "portfolio_turnover",
    "kelly_binary",
    "kelly_objective",
    "sample_covariance",
    "sample_correlation",
    "ewma_cov_update",
    "realized_volatility_hf",
    "shrinkage_covariance",
    "skewness",
    "excess_kurtosis",
    "active_share",
    "value_at_risk_historical",
    "value_at_risk_parametric",
    "expected_shortfall",
    "max_drawdown",
    "time_under_water",
    "stress_pnl",
    "tail_risk_measure",
    "marginal_var",
    "component_var",
]


def _dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return sum(x * y for x, y in zip(a, b))


def _matvec(m: list[list[float]], v: list[float]) -> list[float]:
    if any(len(row) != len(v) for row in m):
        raise ValueError("length mismatch")
    return [_dot(row, v) for row in m]


def _quad(weights: list[float], cov: list[list[float]]) -> float:
    if len(cov) != len(weights):
        raise ValueError("length mismatch")
    return _dot(weights, _matvec(cov, weights))


def _mean(xs: list[float]) -> float:
    if len(xs) == 0:
        raise ValueError("empty series")
    return sum(xs) / len(xs)


def mean_variance_objective(
    weights: list[float],
    expected_returns: list[float],
    cov: list[list[float]],
    risk_aversion: float,
) -> float:
    """w'mu - (risk_aversion / 2) w'Sw."""
    return _dot(weights, expected_returns) - risk_aversion / 2 * _quad(weights, cov)


def unconstrained_mv_weights(
    expected_returns: list[float],
    inv_cov: list[list[float]],
    risk_aversion: float,
) -> list[float]:
    """inv(S) mu / risk_aversion."""
    if risk_aversion <= 0:
        raise ValueError("nonpositive risk aversion")
    return [x / risk_aversion for x in _matvec(inv_cov, expected_returns)]


def minimum_variance_objective(
    weights: list[float], cov: list[list[float]]
) -> float:
    """w'Sw."""
    return _quad(weights, cov)


def portfolio_variance(
    weights: list[float], cov: list[list[float]]
) -> float:
    """w'Sw."""
    return _quad(weights, cov)


def marginal_risk_contribution(
    weights: list[float], cov: list[list[float]]
) -> list[float]:
    """Sw / sqrt(w'Sw)."""
    var = _quad(weights, cov)
    if var <= 0:
        raise ValueError("zero variance")
    vol = math.sqrt(var)
    return [x / vol for x in _matvec(cov, weights)]


def component_risk_contribution(
    weights: list[float], cov: list[list[float]]
) -> list[float]:
    """w * (Sw / sqrt(w'Sw)); sums to portfolio volatility."""
    return [
        w * m for w, m in zip(weights, marginal_risk_contribution(weights, cov))
    ]


def systematic_variance(
    weights: list[float],
    loadings: list[list[float]],
    factor_cov: list[list[float]],
) -> float:
    """e'Fe with factor exposures e = B'w."""
    if len(loadings) != len(weights):
        raise ValueError("length mismatch")
    n_factors = len(factor_cov)
    if any(len(row) != n_factors for row in factor_cov):
        raise ValueError("shape mismatch")
    if any(len(row) != n_factors for row in loadings):
        raise ValueError("length mismatch")
    exposures = [_dot(weights, [row[j] for row in loadings]) for j in range(len(factor_cov))]
    return _dot(exposures, _matvec(factor_cov, exposures))


def garch_update(
    omega: float, alpha: float, beta: float, prev_var: float, shock: float
) -> float:
    """omega + alpha shock^2 + beta prev_var."""
    if omega < 0:
        raise ValueError("negative omega")
    if alpha < 0:
        raise ValueError("negative alpha")
    if beta < 0:
        raise ValueError("negative beta")
    if prev_var < 0:
        raise ValueError("negative prev_var")
    return omega + alpha * shock * shock + beta * prev_var


def garch_variance(
    returns: list[float],
    omega: float,
    alpha: float,
    beta: float,
    initial_var: float,
) -> list[float]:
    """Iterate garch_update over returns with zero-mean shocks."""
    out = []
    var = initial_var
    for r in returns:
        var = garch_update(omega, alpha, beta, var, r)
        out.append(var)
    return out


def volatility_target_scale(
    target_vol: float, forecast_vol: float, cap: float
) -> float:
    """min(target_vol / forecast_vol, cap)."""
    if target_vol <= 0:
        raise ValueError("nonpositive target vol")
    if forecast_vol <= 0:
        raise ValueError("nonpositive forecast vol")
    if cap <= 0:
        raise ValueError("nonpositive cap")
    return min(target_vol / forecast_vol, cap)


def gross_exposure(weights: list[float]) -> float:
    """sum |w|."""
    return sum(abs(w) for w in weights)


def net_exposure(weights: list[float]) -> float:
    """sum w."""
    return sum(weights)


def beta_adjusted_net(weights: list[float], betas: list[float]) -> float:
    """sum w beta."""
    return _dot(weights, betas)


def equal_weights(count: int) -> list[float]:
    """1 / count per asset."""
    if count < 1:
        raise ValueError("nonpositive count")
    return [1 / count for _ in range(count)]


def cap_weights(weights: list[float], cap: float) -> list[float]:
    """Iterative water-filling cap; long-only, budget preserved."""
    if cap <= 0:
        raise ValueError("nonpositive cap")
    if len(weights) == 0:
        raise ValueError("empty series")
    if any(w < 0 for w in weights):
        raise ValueError("negative weight")
    budget = sum(weights)
    if budget == 0:
        return [0] * len(weights)
    if cap * len(weights) < budget:
        raise ValueError("infeasible cap for budget")
    tolerance = cap * 1e-09
    out = list(weights)
    capped = [False] * len(out)
    for _ in range(len(out) + 1):
        free = [i for i in range(len(out)) if not capped[i]]
        if not free:
            break
        remaining = budget - sum(out[i] for i in range(len(out)) if capped[i])
        total_free = sum(out[i] for i in free)
        if total_free == 0:
            share = remaining / len(free)
            for i in free:
                out[i] = share
        else:
            for i in free:
                out[i] = out[i] / total_free * remaining
        over = [i for i in free if out[i] > cap + tolerance]
        if not over:
            break
        for i in over:
            out[i] = cap
            capped[i] = True
    return [min(w, cap) for w in out]


def score_weights(scores: list[float]) -> list[float]:
    """s / sum(s)."""
    total = sum(scores)
    if total == 0:
        raise ValueError("zero total")
    return [s / total for s in scores]


def inverse_vol_weights(vols: list[float]) -> list[float]:
    """(1 / v) / sum(1 / v)."""
    if len(vols) == 0:
        raise ValueError("empty series")
    if any(v <= 0 for v in vols):
        raise ValueError("nonpositive vol")
    inv = [1 / v for v in vols]
    total = sum(inv)
    return [x / total for x in inv]


def diversification_ratio(
    weights: list[float], vols: list[float], cov: list[list[float]]
) -> float:
    """sum(w v) / sqrt(w'Sw)."""
    if len(weights) != len(vols):
        raise ValueError("length mismatch")
    var = _quad(weights, cov)
    if var <= 0:
        raise ValueError("zero variance")
    return _dot(weights, vols) / math.sqrt(var)


def tracking_error(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    annualization: float,
) -> float:
    """std(port - bench) * annualization (population std)."""
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("length mismatch")
    if len(portfolio_returns) == 0:
        raise ValueError("empty series")
    active = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    mean = _mean(active)
    var = sum((a - mean) * (a - mean) for a in active) / len(active)
    return math.sqrt(var) * annualization


def active_weights(
    portfolio_weights: list[float], benchmark_weights: list[float]
) -> list[float]:
    """w_port - w_bench."""
    if len(portfolio_weights) != len(benchmark_weights):
        raise ValueError("length mismatch")
    if len(portfolio_weights) == 0:
        raise ValueError("empty series")
    return [p - b for p, b in zip(portfolio_weights, benchmark_weights)]


def portfolio_turnover(
    new_weights: list[float], old_weights: list[float]
) -> float:
    """sum |new - old| / 2."""
    if len(new_weights) != len(old_weights):
        raise ValueError("length mismatch")
    if len(new_weights) == 0:
        raise ValueError("empty series")
    return sum(abs(n - o) for n, o in zip(new_weights, old_weights)) / 2


def kelly_binary(prob: float, win: float, loss: float) -> float:
    """prob / loss - (1 - prob) / win."""
    if not 0 <= prob <= 1:
        raise ValueError("prob out of range")
    if win <= 0:
        raise ValueError("nonpositive win")
    if loss <= 0:
        raise ValueError("nonpositive loss")
    return prob / loss - (1 - prob) / win


def kelly_objective(
    fraction: float, prob: float, win: float, loss: float
) -> float:
    """prob log(1 + f win) + (1 - prob) log(1 - f loss)."""
    if 1 + fraction * win <= 0:
        raise ValueError("nonpositive 1 + fraction * win")
    if 1 - fraction * loss <= 0:
        raise ValueError("nonpositive 1 - fraction * loss")
    return prob * math.log(1 + fraction * win) + (1 - prob) * math.log(
        1 - fraction * loss
    )


def sample_covariance(
    series_a: list[float], series_b: list[float], ddof: int
) -> float:
    """sum((a - ma)(b - mb)) / (n - ddof)."""
    if len(series_a) != len(series_b):
        raise ValueError("length mismatch")
    if len(series_a) - ddof < 1:
        raise ValueError("nonpositive degrees of freedom")
    mean_a = _mean(series_a)
    mean_b = _mean(series_b)
    return sum(
        (a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b)
    ) / (len(series_a) - ddof)


def sample_correlation(
    series_a: list[float], series_b: list[float]
) -> float:
    """cov / (std_a std_b)."""
    if len(series_a) != len(series_b):
        raise ValueError("length mismatch")
    if len(series_a) == 0:
        raise ValueError("empty series")
    mean_a = _mean(series_a)
    mean_b = _mean(series_b)
    dev_a = [a - mean_a for a in series_a]
    dev_b = [b - mean_b for b in series_b]
    denom = _dot(dev_a, dev_a) * _dot(dev_b, dev_b)
    if denom == 0:
        raise ValueError("zero variance")
    return _dot(dev_a, dev_b) / math.sqrt(denom)


def ewma_cov_update(
    prev_cov: list[list[float]], asset_returns: list[float], decay: float
) -> list[list[float]]:
    """decay prev + (1 - decay) r r'."""
    if len(asset_returns) == 0 or len(prev_cov) == 0:
        raise ValueError("empty series")
    if not 0 <= decay <= 1:
        raise ValueError("decay out of range")
    n = len(asset_returns)
    if len(prev_cov) != n:
        raise ValueError("shape mismatch")
    if any(len(row) != n for row in prev_cov):
        raise ValueError("shape mismatch")
    return [
        [
            decay * p + (1 - decay) * ri * rj
            for p, rj in zip(row, asset_returns)
        ]
        for row, ri in zip(prev_cov, asset_returns)
    ]


def realized_volatility_hf(
    returns: list[float], annualization: float
) -> float:
    """sqrt(sum r^2) * annualization."""
    if len(returns) == 0:
        raise ValueError("empty series")
    return math.sqrt(sum(r * r for r in returns)) * annualization


def shrinkage_covariance(
    sample_cov: list[list[float]], target: list[list[float]], delta: float
) -> list[list[float]]:
    """delta target + (1 - delta) sample, elementwise."""
    if not 0 <= delta <= 1:
        raise ValueError("delta out of range")
    if len(sample_cov) != len(target) or any(
        len(s_row) != len(t_row) for s_row, t_row in zip(sample_cov, target)
    ):
        raise ValueError("shape mismatch")
    return [
        [delta * t + (1 - delta) * s for s, t in zip(s_row, t_row)]
        for s_row, t_row in zip(sample_cov, target)
    ]


def skewness(returns: list[float]) -> float:
    """m3 / (m2 sqrt(m2))."""
    mean = _mean(returns)
    dev = [r - mean for r in returns]
    m2 = sum(d * d for d in dev) / len(dev)
    if m2 == 0:
        raise ValueError("zero variance")
    m3 = sum(d * d * d for d in dev) / len(dev)
    return m3 / (m2 * math.sqrt(m2))


def excess_kurtosis(returns: list[float]) -> float:
    """m4 / m2^2 minus the normal kurtosis (1 + 2)."""
    mean = _mean(returns)
    dev = [r - mean for r in returns]
    m2 = sum(d * d for d in dev) / len(dev)
    if m2 == 0:
        raise ValueError("zero variance")
    m4 = sum(d * d * d * d for d in dev) / len(dev)
    return m4 / (m2 * m2) - (1 + 2)


def active_share(
    portfolio_weights: list[float], benchmark_weights: list[float]
) -> float:
    """sum |w_port - w_bench| / 2."""
    if len(portfolio_weights) != len(benchmark_weights):
        raise ValueError("length mismatch")
    if len(portfolio_weights) == 0:
        raise ValueError("empty series")
    return sum(abs(p - b) for p, b in zip(portfolio_weights, benchmark_weights)) / 2


def value_at_risk_historical(
    returns: list[float], confidence: float
) -> float:
    """-(1 - confidence) quantile of returns."""
    if len(returns) == 0:
        raise ValueError("empty series")
    if not 0 < confidence < 1:
        raise ValueError("confidence not in (0, 1)")
    ordered = sorted(returns)
    k = min(int((1 - confidence) * len(ordered)), len(ordered) - 1)
    return -ordered[k]


def value_at_risk_parametric(returns: list[float], z_score: float) -> float:
    """z_score std - mean."""
    mean = _mean(returns)
    dev = [r - mean for r in returns]
    std = math.sqrt(_dot(dev, dev) / len(dev))
    return z_score * std - mean


def expected_shortfall(returns: list[float], confidence: float) -> float:
    """-mean(returns at or below the VaR quantile); negative means gain (no loss)."""
    cutoff = -value_at_risk_historical(returns, confidence)
    tail = [r for r in returns if r <= cutoff]
    return -_mean(tail)


def max_drawdown(equity: list[float]) -> float:
    """max (peak - equity) / peak."""
    if len(equity) == 0:
        raise ValueError("empty series")
    peak = equity[0]
    worst = 0
    for x in equity:
        peak = max(peak, x)
        dd = (peak - x) / peak if peak != 0 else 0
        worst = max(worst, dd)
    return worst


def time_under_water(equity: list[float]) -> int:
    """Longest run of periods below the running peak."""
    if len(equity) == 0:
        raise ValueError("empty series")
    peak = equity[0]
    run = 0
    longest = 0
    for x in equity:
        if x > peak:
            peak = x
            run = 0
        elif x < peak:
            run = run + 1
            longest = max(longest, run)
    return longest


def stress_pnl(weights: list[float], shocks: list[float]) -> float:
    """sum w shock."""
    if len(weights) == 0:
        raise ValueError("empty series")
    return _dot(weights, shocks)


def tail_risk_measure(cutoffs: list[float], shortfalls: list[float]) -> float:
    """mean(shortfall - cutoff)."""
    if len(cutoffs) != len(shortfalls):
        raise ValueError("length mismatch")
    if len(shortfalls) == 0:
        raise ValueError("empty series")
    return sum(s - c for s, c in zip(shortfalls, cutoffs)) / len(shortfalls)


def marginal_var(
    weights: list[float], cov: list[list[float]], z_score: float
) -> list[float]:
    """z_score Sw / sqrt(w'Sw)."""
    var = _quad(weights, cov)
    if var <= 0:
        raise ValueError("zero variance")
    vol = math.sqrt(var)
    return [z_score * x / vol for x in _matvec(cov, weights)]


def component_var(
    weights: list[float], cov: list[list[float]], z_score: float
) -> list[float]:
    """w * marginal_var; sums to parametric portfolio VaR."""
    return [w * m for w, m in zip(weights, marginal_var(weights, cov, z_score))]
