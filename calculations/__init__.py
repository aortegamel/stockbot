"""Equity calculation package: pure parameterized functions, stdlib only."""

from calculations import (
    valuation,
    fundamentals,
    factors,
    technicals,
    portfolio_risk,
    execution,
    performance,
)
from calculations.valuation import *
from calculations.fundamentals import *
from calculations.factors import *
from calculations.technicals import *
from calculations.portfolio_risk import *
from calculations.execution import *
from calculations.performance import *

__all__ = list(
    dict.fromkeys(
        valuation.__all__
        + fundamentals.__all__
        + factors.__all__
        + technicals.__all__
        + portfolio_risk.__all__
        + execution.__all__
        + performance.__all__
    )
)
