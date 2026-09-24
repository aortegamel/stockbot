"""Fundamentals: profitability, leverage, efficiency, growth, quality.

Formulas (one line each):
roe: net_income / equity
roa: net_income / assets
gross_profitability: gross_profit / assets
cash_roa: operating_cash_flow / assets
total_accruals: net_income - operating_cash_flow
accrual_ratio: (net_income - operating_cash_flow) / assets
cfo_margin: operating_cash_flow / revenue
fcf_margin: fcf / revenue
asset_growth: (assets_now - assets_prior) / assets_prior
investment_measure: capex / assets
net_issuance: (shares_now - shares_prior) / shares_prior
gross_margin: gross_profit / revenue
operating_margin: operating_income / revenue
ebitda_margin: ebitda / revenue
net_margin: net_income / revenue
roce: operating_income / capital_employed
net_debt: total_debt - cash
debt_to_equity: total_debt / equity
net_debt_to_ebitda: net_debt / ebitda
interest_coverage: ebit / interest_expense
current_ratio: current_assets / current_liabilities
quick_ratio: (current_assets - inventory) / current_liabilities
cash_ratio: cash / current_liabilities
asset_turnover: revenue / assets
inventory_turnover: cogs / inventory
receivables_turnover: revenue / receivables
dso: receivables / revenue * days_per_year
cash_conversion_cycle: dso + dio - dpo
beneish_m_score: intercept + w_dsri * dsri + ... + w_tata * tata
cash_conversion: operating_cash_flow / operating_income
margin_change: margin_now - margin_prior
roa_change: roa_now - roa_prior
turnover_change: turnover_now - turnover_prior
buyback_yield: buybacks / market_cap
shareholder_yield: (dividends + buybacks) / market_cap
insider_ownership: insider_shares / shares_outstanding
"""

__all__ = [
    "roe", "roa", "gross_profitability", "cash_roa", "total_accruals",
    "accrual_ratio", "cfo_margin", "fcf_margin", "asset_growth",
    "investment_measure", "net_issuance", "gross_margin",
    "operating_margin", "ebitda_margin", "net_margin", "roce",
    "net_debt", "debt_to_equity", "net_debt_to_ebitda",
    "interest_coverage", "current_ratio", "quick_ratio", "cash_ratio",
    "asset_turnover", "inventory_turnover", "receivables_turnover",
    "dso", "cash_conversion_cycle", "beneish_m_score", "cash_conversion",
    "margin_change", "roa_change", "turnover_change", "buyback_yield",
    "shareholder_yield", "insider_ownership",
]


def roe(net_income: float, equity: float) -> float:
    """Return on equity: net_income / equity."""
    if equity == 0:
        raise ValueError("zero equity")
    return net_income / equity


def roa(net_income: float, total_assets: float) -> float:
    """Return on assets: net_income / assets."""
    if total_assets == 0:
        raise ValueError("zero total assets")
    return net_income / total_assets


def gross_profitability(gross_profit: float, total_assets: float) -> float:
    """Novy-Marx profitability: gross_profit / assets."""
    if total_assets == 0:
        raise ValueError("zero total assets")
    return gross_profit / total_assets


def cash_roa(operating_cash_flow: float, total_assets: float) -> float:
    """Cash return on assets: operating_cash_flow / assets."""
    if total_assets == 0:
        raise ValueError("zero total assets")
    return operating_cash_flow / total_assets


def total_accruals(net_income: float, operating_cash_flow: float) -> float:
    """Accrual component of earnings: net_income - operating_cash_flow."""
    return net_income - operating_cash_flow


def accrual_ratio(net_income: float, operating_cash_flow: float, total_assets: float) -> float:
    """Accruals scaled by assets: (net_income - operating_cash_flow) / assets."""
    if total_assets == 0:
        raise ValueError("zero total assets")
    return (net_income - operating_cash_flow) / total_assets


def cfo_margin(operating_cash_flow: float, revenue: float) -> float:
    """Operating cash margin: operating_cash_flow / revenue."""
    if revenue == 0:
        raise ValueError("zero revenue")
    return operating_cash_flow / revenue


def fcf_margin(fcf: float, revenue: float) -> float:
    """Free cash margin: fcf / revenue."""
    if revenue == 0:
        raise ValueError("zero revenue")
    return fcf / revenue


def asset_growth(assets_now: float, assets_prior: float) -> float:
    """Balance-sheet expansion: (assets_now - assets_prior) / assets_prior."""
    if assets_prior == 0:
        raise ValueError("zero prior assets")
    return (assets_now - assets_prior) / assets_prior


def investment_measure(capital_expenditure: float, total_assets: float) -> float:
    """Investment intensity: capex / assets."""
    if total_assets == 0:
        raise ValueError("zero total assets")
    return capital_expenditure / total_assets


def net_issuance(shares_now: float, shares_prior: float) -> float:
    """Net equity issuance rate: (shares_now - shares_prior) / shares_prior."""
    if shares_prior == 0:
        raise ValueError("zero prior shares")
    return (shares_now - shares_prior) / shares_prior


def gross_margin(gross_profit: float, revenue: float) -> float:
    """Gross margin: gross_profit / revenue."""
    if revenue == 0:
        raise ValueError("zero revenue")
    return gross_profit / revenue


def operating_margin(operating_income: float, revenue: float) -> float:
    """Operating margin: operating_income / revenue."""
    if revenue == 0:
        raise ValueError("zero revenue")
    return operating_income / revenue


def ebitda_margin(ebitda: float, revenue: float) -> float:
    """EBITDA margin: ebitda / revenue."""
    if revenue == 0:
        raise ValueError("zero revenue")
    return ebitda / revenue


def net_margin(net_income: float, revenue: float) -> float:
    """Net margin: net_income / revenue."""
    if revenue == 0:
        raise ValueError("zero revenue")
    return net_income / revenue


def roce(operating_income: float, capital_employed: float) -> float:
    """Return on capital employed: operating_income / capital_employed."""
    if capital_employed == 0:
        raise ValueError("zero capital employed")
    return operating_income / capital_employed


def net_debt(total_debt: float, cash: float) -> float:
    """Debt net of cash: total_debt - cash."""
    return total_debt - cash


def debt_to_equity(total_debt: float, total_equity: float) -> float:
    """Leverage: total_debt / equity."""
    if total_equity == 0:
        raise ValueError("zero equity")
    return total_debt / total_equity


def net_debt_to_ebitda(net_debt_value: float, ebitda: float) -> float:
    """Leverage vs cash earnings: net_debt / ebitda."""
    if ebitda == 0:
        raise ValueError("zero ebitda")
    return net_debt_value / ebitda


def interest_coverage(ebit: float, interest_expense: float) -> float:
    """Debt-service cushion: ebit / interest_expense."""
    if interest_expense == 0:
        raise ValueError("zero interest expense")
    return ebit / interest_expense


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """Short-term liquidity: current_assets / current_liabilities."""
    if current_liabilities == 0:
        raise ValueError("zero current liabilities")
    return current_assets / current_liabilities


def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> float:
    """Acid test: (current_assets - inventory) / current_liabilities."""
    if current_liabilities == 0:
        raise ValueError("zero current liabilities")
    return (current_assets - inventory) / current_liabilities


def cash_ratio(cash: float, current_liabilities: float) -> float:
    """Strictest liquidity: cash / current_liabilities."""
    if current_liabilities == 0:
        raise ValueError("zero current liabilities")
    return cash / current_liabilities


def asset_turnover(revenue: float, total_assets: float) -> float:
    """Asset efficiency: revenue / assets."""
    if total_assets == 0:
        raise ValueError("zero total assets")
    return revenue / total_assets


def inventory_turnover(cogs: float, inventory: float) -> float:
    """Stock efficiency: cogs / inventory."""
    if inventory == 0:
        raise ValueError("zero inventory")
    return cogs / inventory


def receivables_turnover(revenue: float, receivables: float) -> float:
    """Collection efficiency: revenue / receivables."""
    if receivables == 0:
        raise ValueError("zero receivables")
    return revenue / receivables


def dso(receivables: float, revenue: float, days_per_year: float) -> float:
    """Days sales outstanding: receivables / revenue * days_per_year."""
    if revenue == 0:
        raise ValueError("zero revenue")
    if days_per_year <= 0:
        raise ValueError("nonpositive days per year")
    return receivables / revenue * days_per_year


def cash_conversion_cycle(days_sales_outstanding: float, days_inventory_outstanding: float, days_payable_outstanding: float) -> float:
    """Cash tied up in days: dso + dio - dpo."""
    return days_sales_outstanding + days_inventory_outstanding - days_payable_outstanding


def beneish_m_score(dsri: float, gmi: float, aqi: float, sgi: float, depi: float, sgai: float, lvgi: float, tata: float, intercept: float, w_dsri: float, w_gmi: float, w_aqi: float, w_sgi: float, w_depi: float, w_sgai: float, w_lvgi: float, w_tata: float) -> float:
    """Manipulation score: intercept + sum of each weighted 8-variable component."""
    return intercept + w_dsri * dsri + w_gmi * gmi + w_aqi * aqi + w_sgi * sgi + w_depi * depi + w_sgai * sgai + w_lvgi * lvgi + w_tata * tata


def cash_conversion(operating_cash_flow: float, operating_income: float) -> float:
    """Earnings backed by cash: operating_cash_flow / operating_income."""
    if operating_income == 0:
        raise ValueError("zero operating income")
    return operating_cash_flow / operating_income


def margin_change(margin_now: float, margin_prior: float) -> float:
    """Margin momentum: margin_now - margin_prior."""
    return margin_now - margin_prior


def roa_change(roa_now: float, roa_prior: float) -> float:
    """Profitability momentum: roa_now - roa_prior."""
    return roa_now - roa_prior


def turnover_change(turnover_now: float, turnover_prior: float) -> float:
    """Efficiency momentum: turnover_now - turnover_prior."""
    return turnover_now - turnover_prior


def buyback_yield(buybacks: float, market_cap: float) -> float:
    """Repurchase yield: buybacks / market_cap."""
    if market_cap == 0:
        raise ValueError("zero market cap")
    return buybacks / market_cap


def shareholder_yield(dividends: float, buybacks: float, market_cap: float) -> float:
    """Total payout yield: (dividends + buybacks) / market_cap."""
    if market_cap == 0:
        raise ValueError("zero market cap")
    return (dividends + buybacks) / market_cap


def insider_ownership(insider_shares: float, shares_outstanding: float) -> float:
    """Insider stake: insider_shares / shares_outstanding."""
    if shares_outstanding == 0:
        raise ValueError("zero shares outstanding")
    return insider_shares / shares_outstanding
