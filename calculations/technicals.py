"""Technicals: returns, trend, momentum, volatility, volume, and breadth.

Formulas (one line each):
simple_return: (price_now minus price_prior) over price_prior
log_return: natural log of price_now over price_prior
excess_return: asset_return minus benchmark_return
cumulative_return: product of (one plus each return) minus one
annualized_return: (one plus total_return) to (periods_per_year over num_periods) minus one
sma: mean of the last window prices
ema: alpha times newest plus (one minus alpha) times prior average
macd: fast average minus slow average with signal average and histogram
macd_signal: exponential average of a macd line over the signal window
rate_of_change: (price_now minus price_ago) over price_ago times scale
    rsi: scale minus scale over (one plus avg gain over avg loss); flat returns half scale
stochastic_k: (close minus lowest low) over (highest high minus lowest low) times scale
stochastic_d: mean of the last window k values
bollinger_bands: middle average plus or minus num_std times deviation of the last window
bandwidth: (upper minus lower) over middle times scale
true_range: max of high minus low, abs high minus prior close, abs low minus prior close
average_true_range: mean of the last period true ranges
parkinson_variance: mean of squared log high over low scaled by log two
    garman_klass_variance: mean of half squared log high over low minus drift term times squared log close over open, floored at zero
on_balance_volume: cumulative signed volume by close direction
money_flow_index: scale minus scale over (one plus positive over negative flow)
accumulation_distribution: cumulative money-flow multiplier times volume
high_52w_proximity: last price over max of the last lookback times scale
overnight_return: (open minus prior close) over prior close
intraday_return: (close minus open) over open
gap: (open minus prior close) over prior close
advance_decline_line: cumulative advances minus declines
breadth_ratio: advances over declines
new_high_low_ratio: new highs over new lows
signed_trade_imbalance: (buy minus sell) over (buy plus sell)
"""

import math

__all__ = [
    "simple_return", "log_return", "excess_return", "cumulative_return",
    "annualized_return", "sma", "ema", "macd", "macd_signal",
    "rate_of_change", "rsi", "stochastic_k", "stochastic_d",
    "bollinger_bands", "bandwidth", "true_range", "average_true_range",
    "parkinson_variance", "garman_klass_variance", "on_balance_volume",
    "money_flow_index", "accumulation_distribution", "high_52w_proximity",
    "overnight_return", "intraday_return", "gap", "advance_decline_line",
    "breadth_ratio", "new_high_low_ratio", "signed_trade_imbalance",
]


def _mean(values: list[float]) -> float:
    """Mean of values: sum over count."""
    if len(values) == 0:
        raise ValueError("empty series")
    total: float = 0
    for value in values:
        total = total + value
    return total / len(values)


def _std(values: list[float], mean: float) -> float:
    """Population deviation: root mean squared distance from mean."""
    if len(values) == 0:
        raise ValueError("empty series")
    total: float = 0
    for value in values:
        total = total + (value - mean) ** 2
    return math.sqrt(total / len(values))


def _ema_series(values: list[float], alpha: float) -> list[float]:
    """Recursive average: alpha times newest plus (one minus alpha) times prior."""
    if len(values) == 0:
        raise ValueError("empty series")
    if alpha <= 0 or alpha > 1:
        raise ValueError("bad alpha")
    out: list[float] = [values[0]]
    carry = 1 - alpha
    for i in range(1, len(values)):
        out.append(alpha * values[i] + carry * out[i - 1])
    return out


def simple_return(price_now: float, price_prior: float) -> float:
    """Simple return: (price_now minus price_prior) over price_prior."""
    if price_prior == 0:
        raise ValueError("zero price_prior")
    return (price_now - price_prior) / price_prior


def log_return(price_now: float, price_prior: float) -> float:
    """Log return: natural log of price_now over price_prior."""
    if price_now <= 0 or price_prior <= 0:
        raise ValueError("nonpositive price")
    return math.log(price_now / price_prior)


def excess_return(asset_return: float, benchmark_return: float) -> float:
    """Excess return: asset_return minus benchmark_return."""
    return asset_return - benchmark_return


def cumulative_return(returns: list[float]) -> float:
    """Compounded return of a return series."""
    if len(returns) == 0:
        raise ValueError("empty series")
    total: float = 1
    for item in returns:
        total = total * (1 + item)
    return total - 1


def annualized_return(total_return: float, num_periods: float, periods_per_year: float) -> float:
    """Yearly equivalent of a total return earned over num_periods."""
    if num_periods <= 0:
        raise ValueError("nonpositive num_periods")
    if periods_per_year <= 0:
        raise ValueError("nonpositive periods_per_year")
    if total_return <= -1:
        raise ValueError("total_return below -100%")
    return (1 + total_return) ** (periods_per_year / num_periods) - 1


def sma(prices: list[float], window: int) -> list[float]:
    """Rolling mean of the last window prices."""
    if len(prices) == 0:
        raise ValueError("empty series")
    if window < 1 or window > len(prices):
        raise ValueError("bad window")
    out: list[float] = []
    for i in range(len(prices)):
        if i + 1 >= window:
            out.append(_mean(prices[i - window + 1: i + 1]))
    return out


def ema(prices: list[float], alpha: float) -> list[float]:
    """Exponential average with weight alpha on each new price."""
    return _ema_series(prices, alpha)


def macd(prices: list[float], fast_window: int, slow_window: int, signal_window: int) -> dict[str, float]:
    """Trend snapshot: fast minus slow average, signal average, histogram."""
    if len(prices) == 0:
        raise ValueError("empty series")
    if fast_window < 1 or slow_window < 1 or signal_window < 1:
        raise ValueError("bad window")
    fast_line = _ema_series(prices, 2 / (fast_window + 1))
    slow_line = _ema_series(prices, 2 / (slow_window + 1))
    macd_line: list[float] = []
    for i in range(len(prices)):
        macd_line.append(fast_line[i] - slow_line[i])
    signal_line = _ema_series(macd_line, 2 / (signal_window + 1))
    last = len(macd_line) - 1
    return {"macd_line": macd_line[last], "signal_line": signal_line[last], "histogram": macd_line[last] - signal_line[last]}


def macd_signal(macd_line: list[float], signal_window: int) -> list[float]:
    """Signal line: exponential average of a macd line."""
    if signal_window < 1:
        raise ValueError("bad window")
    return _ema_series(macd_line, 2 / (signal_window + 1))


def rate_of_change(prices: list[float], lookback: int, scale: float) -> float:
    """Momentum over lookback, times scale."""
    if len(prices) == 0:
        raise ValueError("empty series")
    if lookback < 1 or lookback >= len(prices):
        raise ValueError("bad lookback")
    price_now = prices[len(prices) - 1]
    price_ago = prices[len(prices) - 1 - lookback]
    if price_ago == 0:
        raise ValueError("zero price_ago")
    return (price_now - price_ago) / price_ago * scale


def rsi(gains: list[float], losses: list[float], period: int, scale: float) -> float:
    """Strength of gains versus losses over period, scaled; flat returns half scale."""
    if len(gains) != len(losses):
        raise ValueError("length mismatch")
    if len(gains) == 0:
        raise ValueError("empty series")
    if period < 1 or period > len(gains):
        raise ValueError("bad period")
    avg_gain = _mean(gains[len(gains) - period: len(gains)])
    avg_loss = _mean(losses[len(losses) - period: len(losses)])
    if avg_gain == 0 and avg_loss == 0:
        return scale / 2
    if avg_loss == 0:
        return scale
    strength = avg_gain / avg_loss
    return scale - scale / (1 + strength)


def stochastic_k(close_now: float, lowest_low: float, highest_high: float, scale: float) -> float:
    """Close position inside the recent range, times scale."""
    if highest_high == lowest_low:
        raise ValueError("zero range")
    return (close_now - lowest_low) / (highest_high - lowest_low) * scale


def stochastic_d(k_values: list[float], window: int) -> list[float]:
    """Average of the last window k values."""
    return sma(k_values, window)


def bollinger_bands(prices: list[float], window: int, num_std: float) -> dict[str, float]:
    """Middle average with upper and lower bands num_std deviations away."""
    if len(prices) == 0:
        raise ValueError("empty series")
    if window < 1 or window > len(prices):
        raise ValueError("bad window")
    if num_std < 0:
        raise ValueError("negative num_std")
    recent = prices[len(prices) - window: len(prices)]
    middle = _mean(recent)
    spread = _std(recent, middle)
    return {"middle": middle, "upper": middle + num_std * spread, "lower": middle - num_std * spread}


def bandwidth(upper: float, lower: float, middle: float, scale: float) -> float:
    """Band width relative to the middle, times scale."""
    if middle == 0:
        raise ValueError("zero middle")
    return (upper - lower) / middle * scale


def true_range(high: float, low: float, close_prior: float) -> float:
    """Largest of high-low and gaps to the prior close."""
    if high < low:
        raise ValueError("high below low")
    high_low = high - low
    high_close = abs(high - close_prior)
    low_close = abs(low - close_prior)
    return max(high_low, high_close, low_close)


def average_true_range(true_ranges: list[float], period: int) -> float:
    """Mean of the last period true ranges."""
    if len(true_ranges) == 0:
        raise ValueError("empty series")
    if period < 1 or period > len(true_ranges):
        raise ValueError("bad period")
    return _mean(true_ranges[len(true_ranges) - period: len(true_ranges)])


def parkinson_variance(highs: list[float], lows: list[float]) -> float:
    """Variance from high-low log ranges."""
    if len(highs) != len(lows):
        raise ValueError("length mismatch")
    if len(highs) == 0:
        raise ValueError("empty series")
    for h in highs:
        if h <= 0:
            raise ValueError("nonpositive price")
    for low in lows:
        if low <= 0:
            raise ValueError("nonpositive price")
    total: float = 0
    for i in range(len(highs)):
        total = total + math.log(highs[i] / lows[i]) ** 2
    return total / (2 * 2 * math.log(2) * len(highs))


def garman_klass_variance(opens: list[float], highs: list[float], lows: list[float], closes: list[float]) -> float:
    """Variance from ranges plus open-close drift, floored at zero."""
    if not (len(opens) == len(highs) == len(lows) == len(closes)):
        raise ValueError("length mismatch")
    if len(closes) == 0:
        raise ValueError("empty series")
    for seq in (opens, highs, lows, closes):
        for price in seq:
            if price <= 0:
                raise ValueError("nonpositive price")
    total: float = 0
    for i in range(len(closes)):
        range_part = math.log(highs[i] / lows[i]) ** 2
        drift_part = math.log(closes[i] / opens[i]) ** 2
        total = total + (1 / 2) * range_part - (2 * math.log(2) - 1) * drift_part
    return max(total / len(closes), 0.0)


def on_balance_volume(closes: list[float], volumes: list[float]) -> list[float]:
    """Cumulative volume signed by close direction."""
    if len(closes) != len(volumes):
        raise ValueError("length mismatch")
    if len(closes) == 0:
        raise ValueError("empty series")
    out: list[float] = [volumes[0]]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            out.append(out[i - 1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            out.append(out[i - 1] - volumes[i])
        else:
            out.append(out[i - 1])
    return out


def money_flow_index(highs: list[float], lows: list[float], closes: list[float], volumes: list[float], period: int, scale: float) -> float:
    """Buying versus selling pressure over period, scaled."""
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ValueError("length mismatch")
    if len(closes) == 0:
        raise ValueError("empty series")
    if period < 1 or period >= len(closes):
        raise ValueError("bad period")
    typical: list[float] = []
    for i in range(len(closes)):
        typical.append(_mean([highs[i], lows[i], closes[i]]))
    start = len(closes) - period
    positive: float = 0
    negative: float = 0
    for i in range(start, len(closes)):
        flow = typical[i] * volumes[i]
        if typical[i] > typical[i - 1]:
            positive = positive + flow
        elif typical[i] < typical[i - 1]:
            negative = negative + flow
    if negative == 0:
        return scale
    ratio = positive / negative
    return scale - scale / (1 + ratio)


def accumulation_distribution(highs: list[float], lows: list[float], closes: list[float], volumes: list[float]) -> list[float]:
    """Cumulative money-flow multiplier times volume."""
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ValueError("length mismatch")
    if len(closes) == 0:
        raise ValueError("empty series")
    out: list[float] = []
    running: float = 0
    for i in range(len(closes)):
        span = highs[i] - lows[i]
        if span == 0:
            multiplier = 0
        else:
            multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / span
        running = running + multiplier * volumes[i]
        out.append(running)
    return out


def high_52w_proximity(prices: list[float], lookback: int, scale: float) -> float:
    """Last price versus the lookback high, times scale."""
    if len(prices) == 0:
        raise ValueError("empty series")
    if lookback < 1 or lookback > len(prices):
        raise ValueError("bad lookback")
    recent = prices[len(prices) - lookback: len(prices)]
    peak = max(recent)
    if peak == 0:
        raise ValueError("zero high")
    return prices[len(prices) - 1] / peak * scale


def overnight_return(close_prior: float, open_now: float) -> float:
    """Gap move from prior close to open."""
    if close_prior == 0:
        raise ValueError("zero close_prior")
    return (open_now - close_prior) / close_prior


def intraday_return(open_now: float, close_now: float) -> float:
    """Session move from open to close."""
    if open_now == 0:
        raise ValueError("zero open")
    return (close_now - open_now) / open_now


def gap(open_now: float, close_prior: float) -> float:
    """Proportional gap from prior close to open."""
    if close_prior == 0:
        raise ValueError("zero close_prior")
    return (open_now - close_prior) / close_prior


def advance_decline_line(advances: list[float], declines: list[float]) -> list[float]:
    """Cumulative advances minus declines."""
    if len(advances) != len(declines):
        raise ValueError("length mismatch")
    if len(advances) == 0:
        raise ValueError("empty series")
    out: list[float] = []
    running: float = 0
    for i in range(len(advances)):
        running = running + advances[i] - declines[i]
        out.append(running)
    return out


def breadth_ratio(advances_now: float, declines_now: float) -> float:
    """Advances over declines."""
    if declines_now == 0:
        raise ValueError("zero declines")
    return advances_now / declines_now


def new_high_low_ratio(new_highs: float, new_lows: float) -> float:
    """New highs over new lows."""
    if new_lows == 0:
        raise ValueError("zero new_lows")
    return new_highs / new_lows


def signed_trade_imbalance(buy_volume: float, sell_volume: float) -> float:
    """Net buy volume over total volume."""
    if buy_volume + sell_volume == 0:
        raise ValueError("zero total volume")
    return (buy_volume - sell_volume) / (buy_volume + sell_volume)
