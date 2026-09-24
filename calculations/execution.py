"""Liquidity, execution, stat-arb, shorting, and options-signal math.

Pure parameterized functions, stdlib only. One-line formulas:
quoted_spread: ask - bid.
midpoint: average of bid and ask.
relative_spread: quoted spread over midpoint times scale.
effective_spread: twice signed distance of trade from midpoint times scale.
realized_spread: twice signed distance of trade from later midpoint times scale.
amihud_illiquidity: mean absolute return over dollar volume times scale.
book_depth: resting bid size plus ask size.
book_imbalance: signed size difference over total size.
cumulative_depth: summed size over the first levels of one side.
vwap: volume-weighted mean price.
twap: time-weighted (mean) price.
implementation_shortfall: side-signed executed plus unexecuted gap to decision price, scaled.
sqrt_market_impact: volatility times coeff times quantity share to exponent.
almgren_chriss_cost: permanent plus temporary cost over the horizon.
pair_distance: mean squared pair deviation times scale.
spread_zscore: spread deviation from mean over spread sd.
hedge_spread: price A minus beta times price B.
cointegration_residual: price Y minus alpha minus beta times price X.
event_ar: stock return minus expected return.
event_car: summed abnormal returns over the window.
event_caar: mean CAR across events.
standardized_surprise: actual minus consensus over dispersion.
merger_spread_annualized: deal spread annualized by day-count ratio.
ex_rights_price: share-weighted average of cum and subscription prices.
t_statistic: mean over standard error.
information_coefficient: Pearson correlation of predicted and realized.
cross_section_rank: share of peers at or below value times scale.
winsorize: clip each value into lower and upper limits.
robust_zscore: deviation from median over MAD times scale.
turnover_rate: volume over shares outstanding times scale.
average_daily_volume: mean volume over the trailing window.
participation_rate: order quantity over market volume times scale.
short_interest_pct: short shares over float shares times scale.
days_to_cover: short shares over average daily volume.
utilization: borrowed over lendable quantity times scale.
borrow_cost: notional times rate times days over basis.
put_call_ratio: put volume over call volume.
iv_premium: implied minus realized volatility times scale.
black_scholes_greeks: call/put delta, gamma, vega, theta from norm CDF via erf.
iv_skew: put IV minus call IV times scale.
iv_term_slope: far-minus-near IV over maturity gap.
"""

import math

__all__ = [
    "quoted_spread",
    "midpoint",
    "relative_spread",
    "effective_spread",
    "realized_spread",
    "amihud_illiquidity",
    "book_depth",
    "book_imbalance",
    "cumulative_depth",
    "vwap",
    "twap",
    "implementation_shortfall",
    "sqrt_market_impact",
    "almgren_chriss_cost",
    "pair_distance",
    "spread_zscore",
    "hedge_spread",
    "cointegration_residual",
    "event_ar",
    "event_car",
    "event_caar",
    "standardized_surprise",
    "merger_spread_annualized",
    "ex_rights_price",
    "t_statistic",
    "information_coefficient",
    "cross_section_rank",
    "winsorize",
    "robust_zscore",
    "turnover_rate",
    "average_daily_volume",
    "participation_rate",
    "short_interest_pct",
    "days_to_cover",
    "utilization",
    "borrow_cost",
    "put_call_ratio",
    "iv_premium",
    "black_scholes_greeks",
    "iv_skew",
    "iv_term_slope",
]


def _mean(values: list[float]) -> float:
    if len(values) == 0:
        raise ValueError("empty series")
    return sum(values) / len(values)


def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def quoted_spread(ask: float, bid: float) -> float:
    return ask - bid


def midpoint(ask: float, bid: float) -> float:
    return (ask + bid) / 2


def relative_spread(ask: float, bid: float, scale: float) -> float:
    mid = midpoint(ask, bid)
    if mid == 0:
        raise ValueError("zero midpoint")
    return (ask - bid) / mid * scale


def effective_spread(trade_price: float, mid: float, direction: float, scale: float) -> float:
    if mid == 0:
        raise ValueError("zero midpoint")
    return 2 * direction * (trade_price - mid) / mid * scale


def realized_spread(trade_price: float, later_midpoint: float, direction: float, scale: float) -> float:
    if later_midpoint == 0:
        raise ValueError("zero midpoint")
    return 2 * direction * (trade_price - later_midpoint) / later_midpoint * scale


def amihud_illiquidity(returns: list[float], dollar_volumes: list[float], scale: float) -> float:
    if len(returns) == 0 or len(dollar_volumes) == 0:
        raise ValueError("empty series")
    if len(returns) != len(dollar_volumes):
        raise ValueError("length mismatch")
    if any(dv == 0 for dv in dollar_volumes):
        raise ValueError("zero dollar volume")
    pairs = list(zip(returns, dollar_volumes))
    return sum(abs(r) / dv for r, dv in pairs) / len(pairs) * scale


def book_depth(bid_sizes: list[float], ask_sizes: list[float]) -> float:
    return float(sum(bid_sizes) + sum(ask_sizes))


def book_imbalance(bid_size: float, ask_size: float) -> float:
    if bid_size + ask_size == 0:
        raise ValueError("zero total size")
    return (bid_size - ask_size) / (bid_size + ask_size)


def cumulative_depth(sizes: list[float], levels: int) -> float:
    if len(sizes) == 0:
        raise ValueError("empty series")
    if levels < 1:
        raise ValueError("nonpositive levels")
    if levels > len(sizes):
        raise ValueError("levels above size count")
    return float(sum(sizes[:levels]))


def vwap(prices: list[float], volumes: list[float]) -> float:
    if len(prices) == 0 or len(volumes) == 0:
        raise ValueError("empty series")
    if len(prices) != len(volumes):
        raise ValueError("length mismatch")
    total = sum(volumes)
    if total == 0:
        raise ValueError("zero total volume")
    return sum(p * v for p, v in zip(prices, volumes)) / total


def twap(prices: list[float]) -> float:
    return _mean(prices)


def implementation_shortfall(
    decision_price: float,
    execution_price: float,
    close_price: float,
    executed_qty: float,
    total_qty: float,
    scale: float,
    side: float,
) -> float:
    if total_qty <= 0:
        raise ValueError("nonpositive total quantity")
    if decision_price == 0:
        raise ValueError("zero decision price")
    if not 0 <= executed_qty <= total_qty:
        raise ValueError("executed quantity out of range")
    if side not in (1, -1):
        raise ValueError("bad side")
    return (
        side
        * (
            executed_qty * (execution_price - decision_price)
            + (total_qty - executed_qty) * (close_price - decision_price)
        )
        / (decision_price * total_qty)
        * scale
    )


def sqrt_market_impact(
    volatility: float, quantity: float, adv: float, coeff: float, exponent: float
) -> float:
    if quantity < 0:
        raise ValueError("negative quantity")
    if adv <= 0:
        raise ValueError("nonpositive adv")
    return coeff * volatility * (quantity / adv) ** exponent


def almgren_chriss_cost(quantity: float, horizon: float, gamma: float, eta: float) -> float:
    if horizon <= 0:
        raise ValueError("nonpositive horizon")
    return gamma * quantity * quantity / 2 + eta * quantity * quantity / horizon


def pair_distance(series_a: list[float], series_b: list[float], scale: float) -> float:
    if len(series_a) == 0 or len(series_b) == 0:
        raise ValueError("empty series")
    if len(series_a) != len(series_b):
        raise ValueError("length mismatch")
    pairs = list(zip(series_a, series_b))
    return sum((a - b) * (a - b) for a, b in pairs) / len(pairs) * scale


def spread_zscore(spread: float, mean: float, spread_sd: float) -> float:
    if spread_sd == 0:
        raise ValueError("zero spread sd")
    return (spread - mean) / spread_sd


def hedge_spread(price_a: float, price_b: float, beta: float) -> float:
    return price_a - beta * price_b


def cointegration_residual(price_y: float, price_x: float, alpha: float, beta: float) -> float:
    return price_y - alpha - beta * price_x


def event_ar(stock_return: float, expected_return: float) -> float:
    return stock_return - expected_return


def event_car(abnormal_returns: list[float]) -> float:
    return float(sum(abnormal_returns))


def event_caar(car_values: list[float]) -> float:
    return _mean(car_values)


def standardized_surprise(actual: float, consensus: float, dispersion: float) -> float:
    if dispersion == 0:
        raise ValueError("zero dispersion")
    return (actual - consensus) / dispersion


def merger_spread_annualized(
    offer_price: float, market_price: float, days_to_close: float, days_per_year: float
) -> float:
    if market_price == 0:
        raise ValueError("zero market price")
    if days_to_close <= 0:
        raise ValueError("nonpositive days to close")
    if days_per_year <= 0:
        raise ValueError("nonpositive days per year")
    return (offer_price - market_price) / market_price * days_per_year / days_to_close


def ex_rights_price(
    cum_price: float, subscription_price: float, old_shares: float, new_shares: float
) -> float:
    if old_shares < 0 or new_shares < 0:
        raise ValueError("negative shares")
    if old_shares + new_shares == 0:
        raise ValueError("zero total shares")
    return (old_shares * cum_price + new_shares * subscription_price) / (old_shares + new_shares)


def t_statistic(sample_mean: float, sample_sd: float, count: int) -> float:
    if count < 2:
        raise ValueError("count below 2")
    if sample_sd == 0:
        raise ValueError("zero sample sd")
    return sample_mean / (sample_sd / math.sqrt(count))


def information_coefficient(predicted: list[float], realized: list[float]) -> float:
    if len(predicted) == 0 or len(realized) == 0:
        raise ValueError("empty series")
    if len(predicted) != len(realized):
        raise ValueError("length mismatch")
    mean_p = _mean(predicted)
    mean_r = _mean(realized)
    cov = sum((a - mean_p) * (b - mean_r) for a, b in zip(predicted, realized))
    var_p = sum((a - mean_p) * (a - mean_p) for a in predicted)
    var_r = sum((b - mean_r) * (b - mean_r) for b in realized)
    if var_p <= 0 or var_r <= 0:
        raise ValueError("zero variance")
    return cov / math.sqrt(var_p * var_r)


def cross_section_rank(values: list[float], value: float, scale: float) -> float:
    if len(values) == 0:
        raise ValueError("empty series")
    return sum(1 for v in values if v <= value) / len(values) * scale


def winsorize(values: list[float], lower: float, upper: float) -> list[float]:
    if len(values) == 0:
        raise ValueError("empty series")
    if lower > upper:
        raise ValueError("lower above upper")
    return [lower if v < lower else upper if v > upper else v for v in values]


def robust_zscore(value: float, median: float, mad: float, scale: float) -> float:
    if mad == 0:
        raise ValueError("zero mad")
    return (value - median) / mad * scale


def turnover_rate(volume: float, shares_outstanding: float, scale: float) -> float:
    if shares_outstanding == 0:
        raise ValueError("zero shares outstanding")
    return volume / shares_outstanding * scale


def average_daily_volume(volumes: list[float], window: int) -> float:
    if len(volumes) == 0:
        raise ValueError("empty series")
    if window < 1:
        raise ValueError("nonpositive window")
    tail = volumes[max(len(volumes) - window, 0):]
    return sum(tail) / len(tail)


def participation_rate(order_qty: float, market_volume: float, scale: float) -> float:
    if market_volume == 0:
        raise ValueError("zero market volume")
    return order_qty / market_volume * scale


def short_interest_pct(short_shares: float, float_shares: float, scale: float) -> float:
    if float_shares == 0:
        raise ValueError("zero float shares")
    return short_shares / float_shares * scale


def days_to_cover(short_shares: float, adv: float) -> float:
    if adv == 0:
        raise ValueError("zero adv")
    return short_shares / adv


def utilization(borrowed_qty: float, lendable_qty: float, scale: float) -> float:
    if lendable_qty == 0:
        raise ValueError("zero lendable quantity")
    return borrowed_qty / lendable_qty * scale


def borrow_cost(notional: float, rate: float, days: float, basis: float) -> float:
    if basis == 0:
        raise ValueError("zero basis")
    return notional * rate * days / basis


def put_call_ratio(put_volume: float, call_volume: float) -> float:
    if call_volume == 0:
        raise ValueError("zero call volume")
    return put_volume / call_volume


def iv_premium(implied_vol: float, realized_vol: float, scale: float) -> float:
    return (implied_vol - realized_vol) * scale


def black_scholes_greeks(
    spot: float, strike: float, rate: float, sigma: float, maturity: float, is_call: bool
) -> dict[str, float]:
    if spot <= 0:
        raise ValueError("nonpositive spot")
    if strike <= 0:
        raise ValueError("nonpositive strike")
    if sigma <= 0:
        raise ValueError("nonpositive sigma")
    if maturity <= 0:
        raise ValueError("nonpositive maturity")
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + sigma * sigma / 2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-rate * maturity)
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf = _norm_pdf(d1)
    gamma = pdf / (spot * sigma * sqrt_t)
    vega = spot * pdf * sqrt_t
    decay = (spot * pdf * sigma) / (2 * sqrt_t)
    if is_call:
        return {
            "delta": nd1,
            "gamma": gamma,
            "vega": vega,
            "theta": -decay - rate * strike * disc * nd2,
        }
    return {
        "delta": nd1 - 1,
        "gamma": gamma,
        "vega": vega,
        "theta": -decay + rate * strike * disc * (1 - nd2),
    }


def iv_skew(put_iv: float, call_iv: float, scale: float) -> float:
    return (put_iv - call_iv) * scale


def iv_term_slope(
    near_iv: float, far_iv: float, near_maturity: float, far_maturity: float
) -> float:
    if far_maturity == near_maturity:
        raise ValueError("equal maturities")
    return (far_iv - near_iv) / (far_maturity - near_maturity)
