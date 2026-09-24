"""Performance measurement library: pure parameterized attribution functions.

Formulas (one line each):
- sharpe_ratio: mean excess return over stdev, scaled to caller periods per year
- sortino_ratio: mean excess over target over downside deviation, scaled to caller periods
- treynor_ratio: mean excess return over beta
- information_ratio: mean active return over tracking error, scaled to caller periods
- calmar_ratio: caller annualized return over caller max-drawdown magnitude
- jensen_alpha: portfolio return minus CAPM expected return
- hit_rate: fraction of periods with positive return
- payoff_ratio: mean win over mean loss magnitude
- profit_factor: gross profit over gross loss
- omega_ratio: gains above threshold over losses below threshold
- time_weighted_return: product of one plus each subperiod return minus one
- money_weighted_irr_objective: net present value of cash flows at caller rate
- management_fee: fee rate times aum times period fraction
- incentive_fee: fee rate times gain above high-water mark, floored at zero
- newey_west_variance: HAC variance of the mean with Bartlett weights to caller lag
- deflated_sharpe: normal CDF of null-adjusted sharpe excess over nonnormal se
- brier_score: mean squared gap between forecast probability and binary outcome
- walk_forward_split: rolling train/test index windows of caller lengths and step
- purged_split: train/test index split purging test overlap plus caller embargo
"""

import math

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "treynor_ratio",
    "information_ratio",
    "calmar_ratio",
    "jensen_alpha",
    "hit_rate",
    "payoff_ratio",
    "profit_factor",
    "omega_ratio",
    "time_weighted_return",
    "money_weighted_irr_objective",
    "management_fee",
    "incentive_fee",
    "newey_west_variance",
    "deflated_sharpe",
    "brier_score",
    "walk_forward_split",
    "purged_split",
]


def _mean(xs: list[float]) -> float:
    if len(xs) == 0:
        raise ValueError("empty series")
    return sum(xs) / len(xs)


def _stdev(xs: list[float], mean: float) -> float:
    if len(xs) == 0:
        raise ValueError("empty series")
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _norm_ppf(p: float, lower: float, upper: float, iterations: int) -> float:
    if lower >= upper:
        raise ValueError("invalid ppf bounds")
    if iterations < 1:
        raise ValueError("nonpositive iterations")
    if p <= 0 or p >= 1:
        raise ValueError("probability outside open unit interval")
    lo = lower
    hi = upper
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sharpe_ratio(
    returns: list[float], risk_free_rate: float, periods_per_year: float
) -> float:
    """Mean per-period excess over stdev, scaled to caller periods per year."""
    excess = [r - risk_free_rate for r in returns]
    mean_excess = _mean(excess)
    vol = _stdev(excess, mean_excess)
    if vol == 0:
        raise ValueError("zero volatility")
    return mean_excess / vol * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: list[float], target_return: float, periods_per_year: float
) -> float:
    """Mean excess over target over downside deviation, scaled to caller periods."""
    mean_return = _mean(returns)
    downside = math.sqrt(
        sum(min(0, r - target_return) ** 2 for r in returns) / len(returns)
    )
    if downside == 0:
        raise ValueError("zero downside deviation")
    return (mean_return - target_return) / downside * math.sqrt(periods_per_year)


def treynor_ratio(
    portfolio_return: float, risk_free_rate: float, beta: float
) -> float:
    """Mean excess return over beta."""
    if beta == 0:
        raise ValueError("zero beta")
    return (portfolio_return - risk_free_rate) / beta


def information_ratio(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    periods_per_year: float,
) -> float:
    """Mean active return over tracking error, scaled to caller periods."""
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("length mismatch")
    active = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    mean_active = _mean(active)
    tracking_error = _stdev(active, mean_active)
    if tracking_error == 0:
        raise ValueError("zero tracking error")
    return mean_active / tracking_error * math.sqrt(periods_per_year)


def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Caller annualized return over caller max-drawdown magnitude."""
    if max_drawdown <= 0:
        raise ValueError("nonpositive max drawdown")
    return annualized_return / max_drawdown


def jensen_alpha(
    portfolio_return: float,
    benchmark_return: float,
    risk_free_rate: float,
    beta: float,
) -> float:
    """Portfolio return minus CAPM expected return."""
    return portfolio_return - (
        risk_free_rate + beta * (benchmark_return - risk_free_rate)
    )


def hit_rate(returns: list[float]) -> float:
    """Fraction of periods with positive return."""
    if len(returns) == 0:
        raise ValueError("empty series")
    return sum(1 for r in returns if r > 0) / len(returns)


def payoff_ratio(returns: list[float]) -> float:
    """Mean win over mean loss magnitude."""
    wins = [r for r in returns if r > 0]
    losses = [-r for r in returns if r < 0]
    if len(wins) == 0:
        raise ValueError("no winning periods")
    if len(losses) == 0:
        raise ValueError("no losing periods")
    return _mean(wins) / _mean(losses)


def profit_factor(returns: list[float]) -> float:
    """Gross profit over gross loss."""
    if len(returns) == 0:
        raise ValueError("empty series")
    gross_loss = sum(-r for r in returns if r < 0)
    if gross_loss == 0:
        raise ValueError("zero gross loss")
    return sum(r for r in returns if r > 0) / gross_loss


def omega_ratio(returns: list[float], threshold: float) -> float:
    """Gains above threshold over losses below threshold."""
    if len(returns) == 0:
        raise ValueError("empty series")
    losses = sum(max(threshold - r, 0) for r in returns)
    if losses == 0:
        raise ValueError("zero below-threshold mass")
    return sum(max(r - threshold, 0) for r in returns) / losses


def time_weighted_return(period_returns: list[float]) -> float:
    """Product of one plus each subperiod return minus one."""
    if len(period_returns) == 0:
        raise ValueError("empty series")
    total = 1
    for r in period_returns:
        total = total * (1 + r)
    return total - 1


def money_weighted_irr_objective(
    cash_flows: list[float], rate: float
) -> float:
    """Net present value of cash flows at caller rate."""
    if len(cash_flows) == 0:
        raise ValueError("empty series")
    if 1 + rate <= 0:
        raise ValueError("nonpositive discount factor")
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))


def management_fee(
    fee_rate: float, assets_under_management: float, period_fraction: float
) -> float:
    """Fee rate times aum times period fraction."""
    return fee_rate * assets_under_management * period_fraction


def incentive_fee(gain: float, fee_rate: float, high_water_mark: float) -> float:
    """Fee rate times gain above high-water mark, floored at zero."""
    return max(gain - high_water_mark, 0) * fee_rate


def newey_west_variance(returns: list[float], lag: int) -> float:
    """HAC variance of the mean with Bartlett weights to caller lag."""
    if len(returns) == 0:
        raise ValueError("empty series")
    if lag < 0:
        raise ValueError("negative lag")
    count = len(returns)
    mean_return = _mean(returns)
    demeaned = [r - mean_return for r in returns]
    var = sum(d * d for d in demeaned) / count
    for lag_index in range(1, lag + 1):
        cov = (
            sum(
                demeaned[t] * demeaned[t - lag_index]
                for t in range(lag_index, count)
            )
            / count
        )
        var = var + 2 * (1 - lag_index / (lag + 1)) * cov
    return var / count


def deflated_sharpe(
    observed_sharpe: float,
    num_observations: int,
    num_trials: int,
    sharpe_std: float,
    skew: float,
    kurtosis: float,
    euler_gamma: float = 0.5772156649015329,
    ppf_lower: float = -10.0,
    ppf_upper: float = 10.0,
    ppf_iterations: int = 100,
) -> float:
    """Normal CDF of null-adjusted sharpe excess over nonnormal se (math-identity defaults overridable)."""
    if num_observations <= 0:
        raise ValueError("nonpositive observations")
    if num_trials < 1:
        raise ValueError("fewer than one trial")
    if num_trials == 1:
        expected_max = 0.0
    else:
        expected_max = (1 - euler_gamma) * _norm_ppf(
            1 - 1 / num_trials, ppf_lower, ppf_upper, ppf_iterations
        ) + euler_gamma * _norm_ppf(
            1 - 1 / (num_trials * math.e), ppf_lower, ppf_upper, ppf_iterations
        )
    null_sharpe = sharpe_std * expected_max
    tail = (kurtosis - 1) / 2
    var = (
        1 - skew * null_sharpe + tail / 2 * null_sharpe * null_sharpe
    ) / num_observations
    if var <= 0:
        raise ValueError("nonpositive sharpe variance")
    return _norm_cdf((observed_sharpe - null_sharpe) / math.sqrt(var))


def brier_score(probabilities: list[float], outcomes: list[float]) -> float:
    """Mean squared gap between forecast probability and binary outcome."""
    if len(probabilities) != len(outcomes):
        raise ValueError("length mismatch")
    if len(probabilities) == 0:
        raise ValueError("empty series")
    for p in probabilities:
        if not 0 <= p <= 1:
            raise ValueError("probability outside unit interval")
    for o in outcomes:
        if o not in (0, 1):
            raise ValueError("nonbinary outcome")
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(
        probabilities
    )


def walk_forward_split(
    num_observations: int, train_length: int, test_length: int, step: int
) -> dict[str, list[list[int]]]:
    """Rolling train/test index windows of caller lengths and step."""
    if num_observations < 1:
        raise ValueError("nonpositive observations")
    if train_length < 1 or test_length < 1 or step < 1:
        raise ValueError("nonpositive window")
    train_windows: list[list[int]] = []
    test_windows: list[list[int]] = []
    start = 0
    while start + train_length + test_length <= num_observations:
        train_windows.append(list(range(start, start + train_length)))
        test_windows.append(
            list(range(start + train_length, start + train_length + test_length))
        )
        start = start + step
    if len(train_windows) == 0:
        raise ValueError("no walk-forward windows")
    return {"train": train_windows, "test": test_windows}


def purged_split(
    event_start_times: list[float],
    event_end_times: list[float],
    test_start: float,
    test_end: float,
    embargo: float,
) -> dict[str, list[int]]:
    """Train/test index split purging test overlap plus caller embargo."""
    if len(event_start_times) != len(event_end_times):
        raise ValueError("length mismatch")
    if len(event_start_times) == 0:
        raise ValueError("empty series")
    if test_start > test_end:
        raise ValueError("test start after test end")
    test = [
        i
        for i in range(len(event_start_times))
        if event_start_times[i] <= test_end and event_end_times[i] >= test_start
    ]
    train = [
        i
        for i in range(len(event_start_times))
        if event_end_times[i] < test_start
        or event_start_times[i] > test_end + embargo
    ]
    return {"train": train, "test": test}
