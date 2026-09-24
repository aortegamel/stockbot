"""Valuation: present-value models, multiples, and return drivers.

Formulas (one line each):
ddm_value: d1 / (r - g)
fcff: ebit * (1 - tax) + dep - capex - d_wc
fcff_value: fcff_next / (wacc - g)
fcfe: fcff - interest * (1 - tax) + net_borrowing
fcfe_value: fcfe_next / (ke - g)
residual_income: eps1 - r * book0
residual_income_value: book0 + ri_next / (r - g)
oj_value: eps1 / r + (eps2 - eps1 - r * (eps1 - dps1)) / (r * (r - gamma)) [domain: 0<=gamma<r, r>0]
pe: price / eps
earnings_yield: eps / price
peg: pe / growth_rate
pb: price / book_per_share
book_to_market: book_per_share / price
ps: price / sales_per_share
enterprise_value: market_cap + debt - cash
ev_ebitda: ev / ebitda
ev_ebit: ev / ebit
ev_sales: ev / sales
sustainable_growth: roe * retention
dupont_roe: margin * turnover * multiplier
dividend_yield: dps / price
payout_ratio: dps / eps
retention_ratio: (eps - dps) / eps
fcf_yield: fcf_per_share / price
cfo_yield: cfo_per_share / price
ev_fcf: ev / fcf
ev_invested_capital: ev / invested_capital
eva: nopat - wacc * capital
roic: nopat / capital
economic_spread: roic - wacc
tobins_q: (equity_mv + debt_mv) / replacement_cost
"""

__all__ = [
    "ddm_value", "fcff", "fcff_value", "fcfe", "fcfe_value",
    "residual_income", "residual_income_value", "oj_value",
    "pe", "earnings_yield", "peg", "pb", "ps",
    "enterprise_value", "ev_ebitda", "ev_ebit", "ev_sales",
    "sustainable_growth", "dupont_roe", "dividend_yield",
    "payout_ratio", "retention_ratio", "fcf_yield", "cfo_yield",
    "ev_fcf", "ev_invested_capital", "eva", "roic",
    "economic_spread", "tobins_q",
]


def ddm_value(dividend_next: float, required_return: float, growth: float) -> float:
    """Gordon constant-growth value: d1 / (r - g)."""
    if required_return <= growth:
        raise ValueError("required return must exceed growth")
    return dividend_next / (required_return - growth)


def fcff(ebit: float, tax_rate: float, depreciation: float, capex: float, change_working_capital: float) -> float:
    """Firm cash flow: ebit * (1 - tax) + dep - capex - d_wc."""
    return ebit * (1 - tax_rate) + depreciation - capex - change_working_capital


def fcff_value(fcff_next: float, wacc: float, growth: float) -> float:
    """Firm value from constant-growth FCFF: fcff_next / (wacc - g)."""
    if wacc <= growth:
        raise ValueError("wacc must exceed growth")
    return fcff_next / (wacc - growth)


def fcfe(fcff_value_in: float, interest_expense: float, tax_rate: float, net_borrowing: float) -> float:
    """Equity cash flow: fcff - interest * (1 - tax) + net_borrowing."""
    return fcff_value_in - interest_expense * (1 - tax_rate) + net_borrowing


def fcfe_value(fcfe_next: float, cost_equity: float, growth: float) -> float:
    """Equity value from constant-growth FCFE: fcfe_next / (ke - g)."""
    if cost_equity <= growth:
        raise ValueError("cost of equity must exceed growth")
    return fcfe_next / (cost_equity - growth)


def residual_income(eps1: float, required_return: float, book_value: float) -> float:
    """Next-period residual income: eps1 - r * book0."""
    return eps1 - required_return * book_value


def residual_income_value(book_value: float, residual_income_next: float, required_return: float, growth: float) -> float:
    """Equity value: book0 + ri_next / (r - g)."""
    if required_return <= growth:
        raise ValueError("required return must exceed growth")
    return book_value + residual_income_next / (required_return - growth)


def oj_value(eps1: float, eps2: float, dividend_next: float, required_return: float, gamma: float) -> float:
    """Ohlson-Juettner-Nauroth value with persistence gamma."""
    if required_return <= 0:
        raise ValueError("required return must be positive")
    if gamma < 0 or gamma >= required_return:
        raise ValueError("gamma must satisfy 0 <= gamma < required return")
    return eps1 / required_return + (eps2 - eps1 - required_return * (eps1 - dividend_next)) / (required_return * (required_return - gamma))


def pe(price: float, eps: float) -> float:
    """Price multiple: price / eps."""
    if eps == 0:
        raise ValueError("zero eps")
    return price / eps


def earnings_yield(eps: float, price: float) -> float:
    """Inverse P/E: eps / price."""
    if price == 0:
        raise ValueError("zero price")
    return eps / price


def peg(pe_ratio: float, growth_rate: float) -> float:
    """Growth-adjusted multiple: pe / growth_rate."""
    if growth_rate == 0:
        raise ValueError("zero growth rate")
    return pe_ratio / growth_rate


def pb(price: float, book_per_share: float) -> float:
    """Price to book: price / book_per_share."""
    if book_per_share == 0:
        raise ValueError("zero book value")
    return price / book_per_share


# qualified-access only (calculations.valuation.book_to_market) due to factors-wins star-import
def book_to_market(book_per_share: float, price: float) -> float:
    """Inverse P/B: book_per_share / price."""
    if price == 0:
        raise ValueError("zero price")
    return book_per_share / price


def ps(price: float, sales_per_share: float) -> float:
    """Price to sales: price / sales_per_share."""
    if sales_per_share == 0:
        raise ValueError("zero sales")
    return price / sales_per_share


def enterprise_value(market_cap: float, total_debt: float, cash: float) -> float:
    """Firm enterprise value: market_cap + debt - cash."""
    return market_cap + total_debt - cash


def ev_ebitda(enterprise_value_in: float, ebitda: float) -> float:
    """EV multiple: ev / ebitda."""
    if ebitda == 0:
        raise ValueError("zero ebitda")
    return enterprise_value_in / ebitda


def ev_ebit(enterprise_value_in: float, ebit: float) -> float:
    """EV multiple: ev / ebit."""
    if ebit == 0:
        raise ValueError("zero ebit")
    return enterprise_value_in / ebit


def ev_sales(enterprise_value_in: float, sales: float) -> float:
    """EV multiple: ev / sales."""
    if sales == 0:
        raise ValueError("zero sales")
    return enterprise_value_in / sales


def sustainable_growth(roe: float, retention: float) -> float:
    """Constant-growth ceiling: roe * retention."""
    return roe * retention


def dupont_roe(net_margin: float, asset_turnover: float, equity_multiplier: float) -> float:
    """Three-way DuPont: margin * turnover * multiplier."""
    return net_margin * asset_turnover * equity_multiplier


def dividend_yield(dividend_per_share: float, price: float) -> float:
    """Cash yield: dps / price."""
    if price == 0:
        raise ValueError("zero price")
    return dividend_per_share / price


def payout_ratio(dividend_per_share: float, eps: float) -> float:
    """Earnings paid out: dps / eps."""
    if eps == 0:
        raise ValueError("zero eps")
    return dividend_per_share / eps


def retention_ratio(dividend_per_share: float, eps: float) -> float:
    """Earnings retained: (eps - dps) / eps."""
    if eps == 0:
        raise ValueError("zero eps")
    return (eps - dividend_per_share) / eps


def fcf_yield(fcf_per_share: float, price: float) -> float:
    """Cash yield on free cash flow: fcf_per_share / price."""
    if price == 0:
        raise ValueError("zero price")
    return fcf_per_share / price


def cfo_yield(cfo_per_share: float, price: float) -> float:
    """Cash yield on operating cash flow: cfo_per_share / price."""
    if price == 0:
        raise ValueError("zero price")
    return cfo_per_share / price


def ev_fcf(enterprise_value_in: float, fcf: float) -> float:
    """EV multiple: ev / fcf."""
    if fcf == 0:
        raise ValueError("zero fcf")
    return enterprise_value_in / fcf


def ev_invested_capital(enterprise_value_in: float, invested_capital: float) -> float:
    """EV to capital: ev / invested_capital."""
    if invested_capital == 0:
        raise ValueError("zero invested capital")
    return enterprise_value_in / invested_capital


def eva(nopat: float, invested_capital: float, wacc: float) -> float:
    """Economic profit: nopat - wacc * capital."""
    return nopat - wacc * invested_capital


def roic(nopat: float, invested_capital: float) -> float:
    """Return on capital: nopat / capital."""
    if invested_capital == 0:
        raise ValueError("zero invested capital")
    return nopat / invested_capital


def economic_spread(roic_value: float, wacc: float) -> float:
    """Value creation per unit of capital: roic - wacc."""
    return roic_value - wacc


def tobins_q(equity_market_value: float, debt_market_value: float, replacement_cost: float) -> float:
    """Market to replacement cost: (equity_mv + debt_mv) / replacement_cost."""
    if replacement_cost == 0:
        raise ValueError("zero replacement cost")
    return (equity_market_value + debt_market_value) / replacement_cost
