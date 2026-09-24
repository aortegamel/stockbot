"""Equity factor library: pure parameterized cross-sectional factor functions.

Formulas (one line each):
- beta: cov(stock, market) over var(market)
- downside_beta: cov over var using only periods where market trails threshold
- size_log: log of market cap in caller-chosen base
- smb_spread: small-cap leg return minus large-cap leg return
- book_to_market: book equity divided by market cap
- hml_spread: value leg return minus growth leg return
- operating_profitability: operating profit divided by book equity
- rmw_spread: robust-profitability leg return minus weak leg return
- investment_rate: asset change over prior assets
- cma_spread: conservative-investment leg return minus aggressive leg return
- momentum_12_2: long-window cumulative return minus recent skip-window return
- short_reversal_score: negated prior return over caller scale
- long_reversal_score: negated long-term return over caller scale
- realized_volatility: return stdev scaled to caller periods per year
- idiosyncratic_volatility: residual stdev scaled to caller periods per year
- turnover_liquidity: mean volume over shares out, scaled to caller periods
- standardized_unexpected_earnings: eps surprise over caller dispersion scale
- piotroski_f_score: sum of nine binary quality signals
- factor_expected_return: dot product of betas and factor premia
- residual_momentum: cumulative residual return over caller scale
- industry_zscore: value minus industry mean over industry std
- composite_alpha: dot product of factor scores and caller weights
- eps_revision: estimate change over caller scale
- earnings_surprise: actual minus consensus over caller scale
- abnormal_return: stock return minus expected return
- cumulative_abnormal_return: sum of abnormal returns
- ml_feature_row: winsorize at caller limits then center and scale per feature
- lasso_objective: sum of squared residuals plus penalty times sum of abs coefs
"""

import math

__all__ = [
    "beta",
    "downside_beta",
    "size_log",
    "smb_spread",
    "book_to_market",
    "hml_spread",
    "operating_profitability",
    "rmw_spread",
    "investment_rate",
    "cma_spread",
    "momentum_12_2",
    "short_reversal_score",
    "long_reversal_score",
    "realized_volatility",
    "idiosyncratic_volatility",
    "turnover_liquidity",
    "standardized_unexpected_earnings",
    "piotroski_f_score",
    "factor_expected_return",
    "residual_momentum",
    "industry_zscore",
    "composite_alpha",
    "eps_revision",
    "earnings_surprise",
    "abnormal_return",
    "cumulative_abnormal_return",
    "ml_feature_row",
    "lasso_objective",
]


def _mean(xs: list[float]) -> float:
    if len(xs) == 0:
        raise ValueError("empty series")
    return sum(xs) / len(xs)


def beta(stock_returns: list[float], market_returns: list[float]) -> float:
    """beta = cov(stock, market) / var(market)."""
    if len(stock_returns) != len(market_returns):
        raise ValueError("length mismatch")
    if len(stock_returns) == 0:
        raise ValueError("empty series")
    mean_stock = _mean(stock_returns)
    mean_market = _mean(market_returns)
    cov = sum(
        (s - mean_stock) * (m - mean_market)
        for s, m in zip(stock_returns, market_returns)
    ) / len(stock_returns)
    var = sum((m - mean_market) ** 2 for m in market_returns) / len(market_returns)
    if var == 0:
        raise ValueError("zero market variance")
    return cov / var


def downside_beta(
    stock_returns: list[float],
    market_returns: list[float],
    threshold_return: float,
) -> float:
    """downside beta = cov / var over periods where market trails threshold."""
    if len(stock_returns) != len(market_returns):
        raise ValueError("length mismatch")
    pairs = [
        (s, m)
        for s, m in zip(stock_returns, market_returns)
        if m < threshold_return
    ]
    if len(pairs) == 0:
        raise ValueError("no downside observations")
    down_stock = [s for s, _ in pairs]
    down_market = [m for _, m in pairs]
    mean_stock = _mean(down_stock)
    mean_market = _mean(down_market)
    cov = sum(
        (s - mean_stock) * (m - mean_market) for s, m in pairs
    ) / len(pairs)
    var = sum((m - mean_market) ** 2 for m in down_market) / len(pairs)
    if var == 0:
        raise ValueError("zero downside market variance")
    return cov / var


def size_log(market_cap: float, log_base: float) -> float:
    """size = log(market cap) in caller-chosen base."""
    if market_cap <= 0:
        raise ValueError("nonpositive market cap")
    if log_base <= 0 or log_base == 1:
        raise ValueError("bad log base")
    return math.log(market_cap) / math.log(log_base)


def smb_spread(small_cap_return: float, large_cap_return: float) -> float:
    """SMB = small-cap leg return minus large-cap leg return."""
    return small_cap_return - large_cap_return


def book_to_market(book_equity: float, market_cap: float) -> float:
    """book-to-market = book equity divided by market cap."""
    if market_cap == 0:
        raise ValueError("zero market cap")
    return book_equity / market_cap


def hml_spread(value_return: float, growth_return: float) -> float:
    """HML = value leg return minus growth leg return."""
    return value_return - growth_return


def operating_profitability(operating_profit: float, book_equity: float) -> float:
    """profitability = operating profit divided by book equity."""
    if book_equity == 0:
        raise ValueError("zero book equity")
    return operating_profit / book_equity


def rmw_spread(robust_return: float, weak_return: float) -> float:
    """RMW = robust-profitability leg return minus weak leg return."""
    return robust_return - weak_return


def investment_rate(total_assets_current: float, total_assets_prior: float) -> float:
    """investment = asset change over prior assets."""
    if total_assets_prior == 0:
        raise ValueError("zero prior assets")
    return (total_assets_current - total_assets_prior) / total_assets_prior


def cma_spread(conservative_return: float, aggressive_return: float) -> float:
    """CMA = conservative-investment leg return minus aggressive leg return."""
    return conservative_return - aggressive_return


def momentum_12_2(
    cum_return_long_window: float, cum_return_recent_window: float
) -> float:
    """momentum = long-window cumulative return minus recent skip-window return."""
    return cum_return_long_window - cum_return_recent_window


def short_reversal_score(prior_return: float, scale: float) -> float:
    """short reversal = negated prior return over caller scale."""
    if scale <= 0:
        raise ValueError("nonpositive scale")
    return -prior_return / scale


def long_reversal_score(long_term_return: float, scale: float) -> float:
    """long reversal = negated long-term return over caller scale."""
    if scale <= 0:
        raise ValueError("nonpositive scale")
    return -long_term_return / scale


def realized_volatility(returns: list[float], periods_per_year: float) -> float:
    """realized vol = return stdev scaled to caller periods per year."""
    if periods_per_year <= 0:
        raise ValueError("nonpositive periods per year")
    mean_return = _mean(returns)
    var = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    return math.sqrt(var * periods_per_year)


def idiosyncratic_volatility(
    residuals: list[float], periods_per_year: float
) -> float:
    """idiosyncratic vol = residual stdev scaled to caller periods per year."""
    if periods_per_year <= 0:
        raise ValueError("nonpositive periods per year")
    mean_residual = _mean(residuals)
    var = sum((r - mean_residual) ** 2 for r in residuals) / len(residuals)
    return math.sqrt(var * periods_per_year)


def turnover_liquidity(
    volumes: list[float], shares_outstanding: float, periods_per_year: float
) -> float:
    """liquidity = mean volume over shares out, scaled to caller periods."""
    if shares_outstanding <= 0:
        raise ValueError("nonpositive shares outstanding")
    if periods_per_year <= 0:
        raise ValueError("nonpositive periods per year")
    return _mean(volumes) / shares_outstanding * periods_per_year


def standardized_unexpected_earnings(
    actual_eps: float, expected_eps: float, surprise_scale: float
) -> float:
    """SUE = eps surprise over caller dispersion scale."""
    if surprise_scale <= 0:
        raise ValueError("nonpositive surprise scale")
    return (actual_eps - expected_eps) / surprise_scale


def piotroski_f_score(
    roa_positive: int,
    cfo_positive: int,
    roa_improvement: int,
    accrual_quality: int,
    leverage_improvement: int,
    liquidity_improvement: int,
    no_dilution: int,
    margin_improvement: int,
    turnover_improvement: int,
) -> int:
    """F-score = sum of nine binary quality signals."""
    signals = (
        roa_positive,
        cfo_positive,
        roa_improvement,
        accrual_quality,
        leverage_improvement,
        liquidity_improvement,
        no_dilution,
        margin_improvement,
        turnover_improvement,
    )
    for s in signals:
        if s not in (0, 1):
            raise ValueError("piotroski signals must be 0 or 1")
    return (
        roa_positive
        + cfo_positive
        + roa_improvement
        + accrual_quality
        + leverage_improvement
        + liquidity_improvement
        + no_dilution
        + margin_improvement
        + turnover_improvement
    )


def factor_expected_return(
    factor_betas: list[float], factor_premia: list[float]
) -> float:
    """expected return = dot product of betas and factor premia."""
    if len(factor_betas) != len(factor_premia):
        raise ValueError("length mismatch")
    return sum(b * p for b, p in zip(factor_betas, factor_premia))


def residual_momentum(cum_residual_return: float, scale: float) -> float:
    """residual momentum = cumulative residual return over caller scale."""
    if scale <= 0:
        raise ValueError("nonpositive scale")
    return cum_residual_return / scale


def industry_zscore(
    stock_value: float, industry_mean: float, industry_std: float
) -> float:
    """industry z-score = value minus industry mean over industry std."""
    if industry_std <= 0:
        raise ValueError("nonpositive industry std")
    return (stock_value - industry_mean) / industry_std


def composite_alpha(
    factor_scores: list[float], factor_weights: list[float]
) -> float:
    """composite alpha = dot product of factor scores and caller weights."""
    if len(factor_scores) != len(factor_weights):
        raise ValueError("length mismatch")
    return sum(s * w for s, w in zip(factor_scores, factor_weights))


def eps_revision(
    new_estimate: float, old_estimate: float, revision_scale: float
) -> float:
    """EPS revision = estimate change over caller scale."""
    if revision_scale <= 0:
        raise ValueError("nonpositive revision scale")
    return (new_estimate - old_estimate) / revision_scale


def earnings_surprise(
    actual_eps: float, consensus_estimate: float, surprise_scale: float
) -> float:
    """earnings surprise = actual minus consensus over caller scale."""
    if surprise_scale <= 0:
        raise ValueError("nonpositive surprise scale")
    return (actual_eps - consensus_estimate) / surprise_scale


def abnormal_return(stock_return: float, expected_return: float) -> float:
    """abnormal return = stock return minus expected return."""
    return stock_return - expected_return


def cumulative_abnormal_return(abnormal_returns: list[float]) -> float:
    """cumulative abnormal return = sum of abnormal returns."""
    if len(abnormal_returns) == 0:
        raise ValueError("empty series")
    return sum(abnormal_returns)


def ml_feature_row(
    raw_features: dict[str, float],
    centers: dict[str, float],
    scales: dict[str, float],
    lower_limits: dict[str, float],
    upper_limits: dict[str, float],
) -> dict[str, float]:
    """standardized row = winsorize at caller limits then center and scale."""
    out: dict[str, float] = {}
    for key, val in raw_features.items():
        if key not in centers or key not in scales:
            raise ValueError(f"missing center/scale for feature: {key}")
        if key not in lower_limits or key not in upper_limits:
            raise ValueError(f"missing limits for feature: {key}")
        if lower_limits[key] > upper_limits[key]:
            raise ValueError(f"inverted limits for feature: {key}")
        clipped = val
        if clipped < lower_limits[key]:
            clipped = lower_limits[key]
        if clipped > upper_limits[key]:
            clipped = upper_limits[key]
        if scales[key] == 0:
            raise ValueError("zero feature scale")
        out[key] = (clipped - centers[key]) / scales[key]
    return out


def lasso_objective(
    residuals: list[float], coefficients: list[float], lambda_penalty: float
) -> float:
    """lasso objective = sum of squared residuals plus penalty over abs coefs."""
    if lambda_penalty < 0:
        raise ValueError("negative lambda penalty")
    return sum(r * r for r in residuals) + lambda_penalty * sum(
        abs(c) for c in coefficients
    )
