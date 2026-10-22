"""Check the parameterized formulas through their public arithmetic interfaces."""

import math
import re
from collections.abc import Callable
from statistics import NormalDist

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from calculations import execution as ex
from calculations import factors as fa
from calculations import fundamentals as fu
from calculations import performance as pe
from calculations import portfolio_risk as pr
from calculations import technicals as te
from calculations import valuation as va


def test_execution_liquidity_depth_and_prices() -> None:
    assert ex.amihud_illiquidity([0.02, -0.06], [100, 300], 1_000_000) == pytest.approx(200)
    assert ex.amihud_illiquidity([0], [10], 1) == 0
    assert ex.average_daily_volume([10, 20, 60], 2) == 40
    assert ex.average_daily_volume([10, 20, 60], 5) == 30
    assert ex.cumulative_depth([10, 20, 60], 2) == 30
    assert ex.cumulative_depth([10, 20, 60], 3) == 90
    assert ex.vwap([10, 12, 20], [2, 3, 0]) == pytest.approx(11.2)
    assert ex.vwap([10, 12], [-1, 3]) == 13
    assert ex.twap([2, 4]) == 3
    assert ex.ex_rights_price(12, 8, 3, 1) == 11
    assert ex.ex_rights_price(12, 8, 0, 1) == 8


def test_execution_costs_events_and_statistics() -> None:
    assert ex.implementation_shortfall(100, 102, 105, 60, 100, 10_000, 1) == 320
    assert ex.implementation_shortfall(100, 102, 105, 60, 100, 10_000, -1) == -320
    assert ex.implementation_shortfall(100, 102, 105, 0, 100, 1, 1) == 0.05
    assert ex.implementation_shortfall(100, 102, 105, 100, 100, 1, 1) == 0.02
    assert ex.implementation_shortfall(100, 102, 105, 60, 100, 0, 1) == 0
    assert ex.implementation_shortfall(-100, -102, -105, 60, 100, 1, 1) == pytest.approx(0.032)
    assert ex.implementation_shortfall(100, 100, 100, 60, 100, 1, 1) == 0
    assert ex.sqrt_market_impact(0.2, 25, 100, 0.5, 0.5) == 0.05
    assert ex.sqrt_market_impact(0.2, 0, 100, 0.5, 0.5) == 0
    assert ex.merger_spread_annualized(110, 100, 90, 360) == pytest.approx(0.4)
    assert ex.pair_distance([1, 4], [2, 2], 10) == 25
    assert ex.pair_distance([1, 4], [1, 4], 10) == 0
    assert ex.t_statistic(2, 4, 16) == 2
    assert ex.information_coefficient([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert ex.information_coefficient([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert ex.information_coefficient([1, 2, 3], [2, 4, 3]) == pytest.approx(0.5)
    assert ex.cross_section_rank([1, 2, 2, 4], 2, 100) == 75
    assert ex.cross_section_rank([1, 2], 0, 100) == 0
    assert ex.winsorize([-3, 0, 1, 5], 0, 2) == [0, 0, 1, 2]
    assert ex.winsorize([-3, 0, 5], 1, 1) == [1, 1, 1]


def test_black_scholes_textbook_greeks() -> None:
    call = ex.black_scholes_greeks(100, 100, 0.05, 0.2, 1, True)
    put = ex.black_scholes_greeks(100, 100, 0.05, 0.2, 1, False)
    assert call == pytest.approx(
        {
            "delta": 0.6368306511756191,
            "gamma": 0.018762017345846895,
            "vega": 37.52403469169379,
            "theta": -6.414027546438197,
        }
    )
    assert put == pytest.approx(
        {
            "delta": -0.3631693488243809,
            "gamma": 0.018762017345846895,
            "vega": 37.52403469169379,
            "theta": -1.657880423934626,
        }
    )


def test_black_scholes_unequal_prices_and_negative_rate() -> None:
    # These inputs make d1 zero, independent of the normal CDF implementation.
    pdf_zero = 1 / math.sqrt(2 * math.pi)
    cdf_d2 = NormalDist().cdf(-0.2)
    call = ex.black_scholes_greeks(100, 100 * math.exp(-0.02), -0.04, 0.2, 1, True)
    put = ex.black_scholes_greeks(100, 100 * math.exp(-0.02), -0.04, 0.2, 1, False)
    assert call == pytest.approx(
        {
            "delta": 0.5,
            "gamma": pdf_zero / 20,
            "vega": 100 * pdf_zero,
            "theta": -10 * pdf_zero + 4 * math.exp(0.02) * cdf_d2,
        }
    )
    assert put == pytest.approx(
        {
            "delta": -0.5,
            "gamma": pdf_zero / 20,
            "vega": 100 * pdf_zero,
            "theta": -10 * pdf_zero - 4 * math.exp(0.02) * (1 - cdf_d2),
        }
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: ex.amihud_illiquidity([], [1], 1), "empty series"),
        (lambda: ex.amihud_illiquidity([1], [], 1), "empty series"),
        (lambda: ex.amihud_illiquidity([1], [1, 2], 1), "length mismatch"),
        (lambda: ex.amihud_illiquidity([1], [0], 1), "zero dollar volume"),
        (lambda: ex.cumulative_depth([], 1), "empty series"),
        (lambda: ex.cumulative_depth([1], 0), "nonpositive levels"),
        (lambda: ex.cumulative_depth([1], 2), "levels above size count"),
        (lambda: ex.vwap([], [1]), "empty series"),
        (lambda: ex.vwap([1], []), "empty series"),
        (lambda: ex.vwap([1], [1, 2]), "length mismatch"),
        (lambda: ex.vwap([1, 2], [1, -1]), "zero total volume"),
        (lambda: ex.implementation_shortfall(0, 1, 1, 0, 0, 1, 0), "nonpositive total quantity"),
        (lambda: ex.implementation_shortfall(0, 1, 1, 0, 1, 1, 1), "zero decision price"),
        (lambda: ex.implementation_shortfall(1, 1, 1, -1, 1, 1, 1), "executed quantity out of range"),
        (lambda: ex.implementation_shortfall(1, 1, 1, 2, 1, 1, 1), "executed quantity out of range"),
        (lambda: ex.implementation_shortfall(1, 1, 1, 1, 1, 1, 0), "bad side"),
        (lambda: ex.sqrt_market_impact(1, -1, 1, 1, 0.5), "negative quantity"),
        (lambda: ex.sqrt_market_impact(1, 1, 0, 1, 0.5), "nonpositive adv"),
        (lambda: ex.pair_distance([], [1], 1), "empty series"),
        (lambda: ex.pair_distance([1], [], 1), "empty series"),
        (lambda: ex.pair_distance([1], [1, 2], 1), "length mismatch"),
        (lambda: ex.merger_spread_annualized(1, 0, 1, 1), "zero market price"),
        (lambda: ex.merger_spread_annualized(1, 1, 0, 1), "nonpositive days to close"),
        (lambda: ex.merger_spread_annualized(1, 1, 1, 0), "nonpositive days per year"),
        (lambda: ex.ex_rights_price(1, 1, -1, 1), "negative shares"),
        (lambda: ex.ex_rights_price(1, 1, 1, -1), "negative shares"),
        (lambda: ex.ex_rights_price(1, 1, 0, 0), "zero total shares"),
        (lambda: ex.t_statistic(1, 1, 1), "count below 2"),
        (lambda: ex.t_statistic(1, 0, 2), "zero sample sd"),
        (lambda: ex.information_coefficient([], [1]), "empty series"),
        (lambda: ex.information_coefficient([1], []), "empty series"),
        (lambda: ex.information_coefficient([1], [1, 2]), "length mismatch"),
        (lambda: ex.information_coefficient([1, 1], [1, 2]), "zero variance"),
        (lambda: ex.information_coefficient([1, 2], [1, 1]), "zero variance"),
        (lambda: ex.cross_section_rank([], 1, 1), "empty series"),
        (lambda: ex.winsorize([], 0, 1), "empty series"),
        (lambda: ex.winsorize([1], 2, 1), "lower above upper"),
        (lambda: ex.average_daily_volume([], 1), "empty series"),
        (lambda: ex.average_daily_volume([1], 0), "nonpositive window"),
        (lambda: ex.black_scholes_greeks(0, 1, 0, 1, 1, True), "nonpositive spot"),
        (lambda: ex.black_scholes_greeks(1, 0, 0, 1, 1, True), "nonpositive strike"),
        (lambda: ex.black_scholes_greeks(1, 1, 0, 0, 1, True), "nonpositive sigma"),
        (lambda: ex.black_scholes_greeks(1, 1, 0, 1, 0, True), "nonpositive maturity"),
    ],
)
def test_execution_errors(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        call()
    assert str(error.value) == message


def test_factor_exposures_and_scores() -> None:
    assert fa.beta([3, 5, 7], [1, 2, 3]) == 2
    assert fa.beta([1, 1, 1], [1, 2, 3]) == 0
    assert fa.downside_beta([-6, -2, 1000, -1000], [-3, -1, 0, 2], 0) == 2
    assert fa.factor_expected_return([1, 2, -1], [0.03, 0.02, 0.01]) == pytest.approx(0.06)
    assert fa.factor_expected_return([], []) == 0
    assert fa.composite_alpha([1, -2, 3], [0.5, 0.25, 0.25]) == 0.75
    assert fa.composite_alpha([], []) == 0
    assert fa.piotroski_f_score(1, 0, 1, 0, 1, 0, 1, 0, 1) == 5
    assert fa.piotroski_f_score(*([0] * 9)) == 0
    assert fa.piotroski_f_score(*([1] * 9)) == 9
    assert fa.lasso_objective([1, -2], [-3, 4], 0.5) == 8.5
    assert fa.lasso_objective([], [], 0) == 0
    assert fa.lasso_objective([1, -2], [100], 0) == 5
    assert fa.size_log(1000, 10) == pytest.approx(3)
    assert fa.size_log(4, 0.5) == pytest.approx(-2)


def test_factor_volatility_and_liquidity() -> None:
    assert fa.realized_volatility([-1, 1], 4) == 2
    assert fa.idiosyncratic_volatility([-2, 0, 2], 6) == 4
    assert fa.realized_volatility([3, 3], 252) == 0
    assert fa.idiosyncratic_volatility([3, 3], 252) == 0
    assert fa.turnover_liquidity([100, 200], 1000, 12) == pytest.approx(1.8)
    assert fu.dso(50, 200, 360) == 90
    assert fu.dso(-50, 200, 360) == -90
    assert va.oj_value(2, 2.5, 1, 0.1, 0.05) == pytest.approx(100)
    assert va.oj_value(2, 2.5, 1, 0.1, 0) == pytest.approx(60)


def test_feature_row_clips_then_centers_and_scales() -> None:
    raw = {"low": -5.0, "mid": 4.0, "high": 20.0, "edge": 0.0}
    centers = dict.fromkeys(raw, 2.0)
    scales = {"low": 2.0, "mid": 2.0, "high": -2.0, "edge": 1.0}
    lower = dict.fromkeys(raw, 0.0)
    upper = dict.fromkeys(raw, 10.0)
    assert fa.ml_feature_row(raw, centers, scales, lower, upper) == {"low": -1, "mid": 1, "high": -4, "edge": -2}
    assert raw == {"low": -5.0, "mid": 4.0, "high": 20.0, "edge": 0.0}
    empty: dict[str, float] = {}
    assert fa.ml_feature_row(empty, empty, empty, empty, empty) == {}


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: fa.beta([1], []), "length mismatch"),
        (lambda: fa.beta([], []), "empty series"),
        (lambda: fa.beta([1, 2], [1, 1]), "zero market variance"),
        (lambda: fa.downside_beta([1], [], 0), "length mismatch"),
        (lambda: fa.downside_beta([1, 2], [0, 1], 0), "no downside observations"),
        (lambda: fa.downside_beta([1, 2], [-1, -1], 0), "zero downside market variance"),
        (lambda: fa.size_log(0, 10), "nonpositive market cap"),
        (lambda: fa.size_log(10, 0), "bad log base"),
        (lambda: fa.size_log(10, 1), "bad log base"),
        (lambda: fa.factor_expected_return([1], []), "length mismatch"),
        (lambda: fa.composite_alpha([1], []), "length mismatch"),
        (lambda: fa.realized_volatility([], 1), "empty series"),
        (lambda: fa.realized_volatility([1], 0), "nonpositive periods per year"),
        (lambda: fa.idiosyncratic_volatility([], 1), "empty series"),
        (lambda: fa.idiosyncratic_volatility([1], 0), "nonpositive periods per year"),
        (lambda: fa.turnover_liquidity([], 1, 1), "empty series"),
        (lambda: fa.turnover_liquidity([1], 0, 1), "nonpositive shares outstanding"),
        (lambda: fa.turnover_liquidity([1], 1, 0), "nonpositive periods per year"),
        (lambda: fa.piotroski_f_score(1, 1, 1, 1, 1, 1, 1, 1, 2), "piotroski signals must be 0 or 1"),
        (lambda: fa.lasso_objective([], [], -1), "negative lambda penalty"),
        (lambda: fa.ml_feature_row({"a": 1}, {}, {"a": 1}, {"a": 0}, {"a": 2}), "missing center/scale for feature: a"),
        (lambda: fa.ml_feature_row({"a": 1}, {"a": 0}, {}, {"a": 0}, {"a": 2}), "missing center/scale for feature: a"),
        (lambda: fa.ml_feature_row({"a": 1}, {"a": 0}, {"a": 1}, {}, {"a": 2}), "missing limits for feature: a"),
        (lambda: fa.ml_feature_row({"a": 1}, {"a": 0}, {"a": 1}, {"a": 0}, {}), "missing limits for feature: a"),
        (lambda: fa.ml_feature_row({"a": 1}, {"a": 0}, {"a": 1}, {"a": 3}, {"a": 2}), "inverted limits for feature: a"),
        (lambda: fa.ml_feature_row({"a": 1}, {"a": 0}, {"a": 0}, {"a": 0}, {"a": 2}), "zero feature scale"),
        (lambda: fu.dso(1, 0, 360), "zero revenue"),
        (lambda: fu.dso(1, 1, 0), "nonpositive days per year"),
        (lambda: va.oj_value(1, 1, 1, 0, 0), "required return must be positive"),
        (lambda: va.oj_value(1, 1, 1, 0.1, -0.1), "gamma must satisfy 0 <= gamma < required return"),
        (lambda: va.oj_value(1, 1, 1, 0.1, 0.1), "gamma must satisfy 0 <= gamma < required return"),
    ],
)
def test_factor_and_valuation_errors(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        call()
    assert str(error.value) == message


def test_performance_ratios_and_compounding() -> None:
    assert pe.sharpe_ratio([0, 2], 0, 4) == 2
    assert pe.sortino_ratio([-1, 3], 0, 4) == pytest.approx(2 * math.sqrt(2))
    assert pe.sharpe_ratio([-0.02, 0.06], 0.01, 4) == pytest.approx(0.5)
    assert pe.sharpe_ratio([-1, 1], 0, 4) == 0
    assert pe.sharpe_ratio([-2, 0], 0, 4) == -2
    assert pe.sharpe_ratio([-1, 1], 0, 0) == 0
    assert pe.sortino_ratio([-2, 2], 1, 4) == pytest.approx(-2 * math.sqrt(2) / 3)
    assert pe.sortino_ratio([-1, 3], 1, 4) == 0
    assert pe.sortino_ratio([-1, 3], 0, 0) == 0
    assert pe.information_ratio([1, 3], [0, 0], 4) == 4
    assert pe.hit_rate([-1, 0, 2, 3]) == 0.5
    assert pe.payoff_ratio([2, 4, -1, -2, 0]) == 2
    assert pe.profit_factor([2, 4, -1, -2, 0]) == 2
    assert pe.profit_factor([-1, -2]) == 0
    assert pe.omega_ratio([0, 2, 4], 1) == 4
    assert pe.time_weighted_return([0.1, -0.1]) == pytest.approx(-0.01)
    assert pe.time_weighted_return([0.1, -1, 0.5]) == -1
    assert pe.time_weighted_return([0, 0]) == 0
    assert pe.time_weighted_return([-0.5]) == -0.5
    assert pe.money_weighted_irr_objective([-100, 110], 0.1) == pytest.approx(0, abs=1e-12)
    assert pe.money_weighted_irr_objective([-100, 60, 60], 0) == 20
    assert pe.newey_west_variance([1, 2, 3, 4], 0) == pytest.approx(0.3125)
    assert pe.newey_west_variance([1, 2, 3, 4], 1) == pytest.approx(0.390625)
    assert pe.newey_west_variance([3, 3], 5) == 0
    assert pe.brier_score([0.25, 0.75], [0, 1]) == 0.0625
    assert pe.brier_score([0, 1], [0, 1]) == 0
    assert pe.brier_score([0, 1], [1, 0]) == 1


def test_deflated_sharpe_against_normal_distribution() -> None:
    normal = NormalDist()
    assert pe.deflated_sharpe(0.2, 100, 1, 0.3, 0, 3) == pytest.approx(normal.cdf(2))
    # Independent normal quantiles check the public consumer of the bisection solver.
    trials = 5
    gamma = 0.5772156649015329
    expected_max = (1 - gamma) * normal.inv_cdf(1 - 1 / trials) + gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
    null = 0.3 * expected_max
    variance = (1 + null * null / 2) / 100
    expected = normal.cdf((0.2 - null) / math.sqrt(variance))
    assert pe.deflated_sharpe(0.2, 100, trials, 0.3, 0, 3) == pytest.approx(expected)
    assert pe.deflated_sharpe(0.2, 100, 20, 0.3, 0, 3) < expected
    nonnormal_variance = (1 - 0.7 * null + (5.5 - 1) * null * null / 4) / 100
    assert pe.deflated_sharpe(0.2, 100, trials, 0.3, 0.7, 5.5) == pytest.approx(
        normal.cdf((0.2 - null) / math.sqrt(nonnormal_variance))
    )
    assert pe.deflated_sharpe(0, 1, 1, 0, 0, 3) == 0.5
    assert pe.deflated_sharpe(-0.2, 100, 1, 0, 0, 3) == pytest.approx(normal.cdf(-2))


def test_temporal_training_splits() -> None:
    assert pe.walk_forward_split(9, 3, 2, 2) == {
        "train": [[0, 1, 2], [2, 3, 4], [4, 5, 6]],
        "test": [[3, 4], [5, 6], [7, 8]],
    }
    assert pe.walk_forward_split(5, 3, 2, 20) == {"train": [[0, 1, 2]], "test": [[3, 4]]}
    assert pe.purged_split([0, 2, 5, 7, 9], [1, 4, 6, 8, 10], 4, 6, 2) == {"train": [0, 4], "test": [1, 2]}
    assert pe.purged_split([0, 2, 5], [1, 3, 6], 2, 2, 0) == {"train": [0, 2], "test": [1]}
    assert pe.walk_forward_split(2, 1, 1, 1) == {"train": [[0]], "test": [[1]]}
    assert pe.purged_split([0, 2, 4, 5, 6], [1, 3, 4, 5, 7], 3, 4, 1) == {
        "train": [0, 4],
        "test": [1, 2],
    }
    assert pe.purged_split([0, 6], [1, 7], 3, 4, 0) == {"train": [0, 1], "test": []}


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: pe.sharpe_ratio([], 0, 1), "empty series"),
        (lambda: pe.sharpe_ratio([1, 1], 0, 1), "zero volatility"),
        (lambda: pe.sortino_ratio([], 0, 1), "empty series"),
        (lambda: pe.sortino_ratio([0, 1], 0, 1), "zero downside deviation"),
        (lambda: pe.information_ratio([1], [], 1), "length mismatch"),
        (lambda: pe.information_ratio([], [], 1), "empty series"),
        (lambda: pe.information_ratio([1, 2], [0, 1], 1), "zero tracking error"),
        (lambda: pe.hit_rate([]), "empty series"),
        (lambda: pe.payoff_ratio([-1, 0]), "no winning periods"),
        (lambda: pe.payoff_ratio([1, 0]), "no losing periods"),
        (lambda: pe.profit_factor([]), "empty series"),
        (lambda: pe.profit_factor([0, 1]), "zero gross loss"),
        (lambda: pe.omega_ratio([], 0), "empty series"),
        (lambda: pe.omega_ratio([1, 2], 1), "zero below-threshold mass"),
        (lambda: pe.time_weighted_return([]), "empty series"),
        (lambda: pe.money_weighted_irr_objective([], 0), "empty series"),
        (lambda: pe.money_weighted_irr_objective([1], -1), "nonpositive discount factor"),
        (lambda: pe.newey_west_variance([], 0), "empty series"),
        (lambda: pe.newey_west_variance([1], -1), "negative lag"),
        (lambda: pe.deflated_sharpe(1, 0, 1, 1, 0, 3), "nonpositive observations"),
        (lambda: pe.deflated_sharpe(1, 1, 0, 1, 0, 3), "fewer than one trial"),
        (lambda: pe.deflated_sharpe(1, 1, 2, 1, 100, 3), "nonpositive sharpe variance"),
        (lambda: pe.deflated_sharpe(1, 1, 2, 1, 0, 3, ppf_lower=1, ppf_upper=1), "invalid ppf bounds"),
        (lambda: pe.deflated_sharpe(1, 1, 2, 1, 0, 3, ppf_iterations=0), "nonpositive iterations"),
        # 1 - 1 / num_trials rounds to 1.0 above 2**53 trials, so the normal quantile sees p = 1.
        (lambda: pe.deflated_sharpe(0.2, 100, 10**17, 0.3, 0, 3), "probability outside open unit interval"),
        (lambda: pe.brier_score([0], []), "length mismatch"),
        (lambda: pe.brier_score([], []), "empty series"),
        (lambda: pe.brier_score([-0.1], [0]), "probability outside unit interval"),
        (lambda: pe.brier_score([1.1], [0]), "probability outside unit interval"),
        (lambda: pe.brier_score([0.5], [0.5]), "nonbinary outcome"),
        (lambda: pe.walk_forward_split(0, 1, 1, 1), "nonpositive observations"),
        (lambda: pe.walk_forward_split(5, 0, 1, 1), "nonpositive window"),
        (lambda: pe.walk_forward_split(5, 1, 0, 1), "nonpositive window"),
        (lambda: pe.walk_forward_split(5, 1, 1, 0), "nonpositive window"),
        (lambda: pe.walk_forward_split(4, 3, 2, 1), "no walk-forward windows"),
        (lambda: pe.purged_split([1], [], 0, 1, 0), "length mismatch"),
        (lambda: pe.purged_split([], [], 0, 1, 0), "empty series"),
        (lambda: pe.purged_split([1], [2], 2, 1, 0), "test start after test end"),
    ],
)
def test_performance_errors(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        call()
    assert str(error.value) == message


def test_portfolio_allocations_and_redistribution() -> None:
    assert pr.equal_weights(4) == [0.25] * 4
    assert pr.equal_weights(1) == [1]
    assert pr.score_weights([1, 2, 1]) == [0.25, 0.5, 0.25]
    assert pr.score_weights([-1, 3]) == [-0.5, 1.5]
    assert pr.inverse_vol_weights([1, 2]) == pytest.approx([2 / 3, 1 / 3])
    weights = [0.8, 0.1, 0.1]
    assert pr.cap_weights(weights, 0.5) == pytest.approx([0.5, 0.25, 0.25])
    assert weights == [0.8, 0.1, 0.1]
    assert pr.cap_weights([0.7, 0.2, 0.1], 0.4) == pytest.approx([0.4, 0.4, 0.2])
    assert pr.cap_weights([1, 0, 0], 0.5) == [0.5, 0.25, 0.25]
    assert pr.cap_weights([3, 1], 2) == [2, 2]
    assert pr.cap_weights([0, 0], 1) == [0, 0]
    assert pr.cap_weights([0.2, 0.3], 0.4) == pytest.approx([0.2, 0.3])
    # Values within the existing cap tolerance only receive the final clip.
    assert pr.cap_weights([0.5 + 1e-10, 0.2], 0.5) == [0.5, 0.2]


def test_portfolio_risk_decomposition() -> None:
    weights = [0.5, 0.5]
    cov = [[4.0, 0.0], [0.0, 9.0]]
    volatility = math.sqrt(3.25)
    assert pr.portfolio_variance(weights, cov) == 3.25
    assert pr.unconstrained_mv_weights([0.1, 0.2], [[2, 0], [0, 3]], 2) == pytest.approx([0.1, 0.3])
    assert pr.unconstrained_mv_weights([0.1, -0.2], [[2, -1], [-1, 3]], 2) == pytest.approx([0.2, -0.35])
    assert pr.unconstrained_mv_weights([0, 0], [[2, -1], [-1, 3]], 2) == [0, 0]
    assert pr.unconstrained_mv_weights([], [], 1) == []
    assert pr.marginal_risk_contribution(weights, cov) == pytest.approx([2 / volatility, 4.5 / volatility])
    assert sum(pr.component_risk_contribution(weights, cov)) == pytest.approx(volatility)
    assert pr.marginal_var(weights, cov, 2) == pytest.approx([4 / volatility, 9 / volatility])
    assert sum(pr.component_var(weights, cov, 2)) == pytest.approx(2 * volatility)
    assert pr.diversification_ratio([0.5, 0.5], [2, 2], [[4, 0], [0, 4]]) == pytest.approx(math.sqrt(2))
    assert pr.systematic_variance([0.25, 0.75], [[1, 2], [3, 4]], [[4, 0], [0, 1]]) == 37.25
    assert pr.volatility_target_scale(0.2, 0.1, 3) == 2
    assert pr.volatility_target_scale(0.2, 0.1, 1) == 1
    assert pr.volatility_target_scale(0.2, 0.1, 2) == 2
    assert pr.garch_update(0.1, 0.2, 0.5, 0.4, -2) == pytest.approx(1.1)
    assert pr.garch_variance([2, 0], 0.1, 0.2, 0.5, 0.4) == pytest.approx([1.1, 0.65])
    assert pr.garch_update(0, 0, 0, 0, 10) == 0
    assert pr.kelly_binary(0.6, 1, 1) == pytest.approx(0.2)
    assert pr.kelly_binary(0, 2, 1) == -0.5
    assert pr.kelly_binary(1, 2, 1) == 1
    assert pr.kelly_objective(0.2, 0.6, 1, 1) == pytest.approx(0.020135513550688863)
    assert pr.kelly_objective(0.2, 0.6, 1, 1) > pr.kelly_objective(0.1, 0.6, 1, 1)
    assert pr.kelly_objective(0.2, 0.6, 1, 1) > pr.kelly_objective(0.3, 0.6, 1, 1)
    assert pr.kelly_binary(0.6, 2, 0.5) == pytest.approx(1)
    assert pr.kelly_binary(0.5, 1, 1) == 0
    assert pr.kelly_objective(0, 0.6, 2, 0.5) == 0
    assert pr.kelly_objective(-0.2, 0.2, 2, 0.5) == pytest.approx(0.2 * math.log(0.6) + 0.8 * math.log(1.1))


def test_portfolio_covariance_updates_and_moments() -> None:
    assert pr.sample_covariance([1, 2, 3], [2, 4, 6], 1) == 2
    assert pr.sample_covariance([1, 2, 3], [2, 4, 6], 0) == pytest.approx(4 / 3)
    assert pr.sample_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert pr.sample_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert pr.sample_correlation([1, 2, 3], [2, 4, 3]) == pytest.approx(0.5)
    previous = [[1.0, 0.0], [0.0, 1.0]]
    assert pr.ewma_cov_update(previous, [2, -1], 0.5) == [[2.5, -1], [-1, 1]]
    assert pr.ewma_cov_update(previous, [2, -1], 0) == [[4, -2], [-2, 1]]
    assert pr.ewma_cov_update(previous, [2, -1], 1) == previous
    assert previous == [[1, 0], [0, 1]]
    assert pr.shrinkage_covariance([[1, 2], [2, 5]], [[4, 0], [0, 4]], 0.25) == [[1.75, 1.5], [1.5, 4.75]]
    assert pr.shrinkage_covariance([[1]], [[4]], 0) == [[1]]
    assert pr.shrinkage_covariance([[1]], [[4]], 1) == [[4]]
    assert pr.shrinkage_covariance([], [], 0.5) == []
    assert pr.realized_volatility_hf([3, 4], 2) == 10
    assert pr.skewness([-1, 0, 1]) == 0
    assert pr.skewness([0, 0, 3]) == pytest.approx(1 / math.sqrt(2))
    assert pr.excess_kurtosis([-1, 1]) == -2


def test_portfolio_losses_and_relative_positions() -> None:
    assert pr.active_weights([0.7, 0.3], [0.4, 0.6]) == pytest.approx([0.3, -0.3])
    assert pr.active_share([0.7, 0.3], [0.4, 0.6]) == pytest.approx(0.3)
    assert pr.portfolio_turnover([0.7, 0.3], [0.4, 0.6]) == pytest.approx(0.3)
    assert pr.tracking_error([1, 3], [0, 0], 2) == 2
    assert pr.tail_risk_measure([1, 2], [3, 6]) == 3
    returns = [-0.1, -0.05, -0.02, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    assert pr.value_at_risk_historical(returns, 0.75) == 0.02
    assert pr.expected_shortfall(returns, 0.75) == pytest.approx(0.17 / 3)
    assert pr.value_at_risk_historical(returns, 0.01) == -0.06
    assert pr.expected_shortfall([0.1, 0.2], 0.5) == pytest.approx(-0.15)
    assert pr.value_at_risk_parametric([-1, 3], 1.5) == 2
    assert pr.value_at_risk_parametric([0.2, 0.2], 3) == -0.2
    assert pr.value_at_risk_parametric([-1, 3], 0) == -1
    assert pr.value_at_risk_parametric([-1, 3], -1) == -3
    assert pr.value_at_risk_historical([0], 0.5) == 0
    assert pr.expected_shortfall([0], 0.5) == 0
    assert pr.expected_shortfall([-0.2, -0.1, -0.1, 0.4], 0.5) == pytest.approx(0.4 / 3)
    assert pr.max_drawdown([100, 120, 90, 130, 65]) == 0.5
    assert pr.max_drawdown([0, 0, 10, 5]) == 0.5
    assert pr.max_drawdown([1, 2, 3]) == 0
    assert pr.max_drawdown([0, 0]) == 0
    assert pr.max_drawdown([100, 0]) == 1
    # Equality with the peak does not reset the existing run counter.
    assert pr.time_under_water([100, 90, 100, 90, 110, 109, 108, 110, 107]) == 3
    assert pr.gross_exposure([0.6, -0.4, 0.2]) == pytest.approx(1.2)
    assert pr.gross_exposure([]) == 0
    assert pr.net_exposure([0.6, -0.4, 0.2]) == pytest.approx(0.4)
    assert pr.net_exposure([]) == 0


def test_valuation_constant_growth_known_answers() -> None:
    assert va.ddm_value(2, 0.1, 0.04) == pytest.approx(200 / 6)
    assert va.fcff_value(100, 0.1, 0.04) == pytest.approx(100 / 0.06)
    assert va.fcfe_value(60, 0.12, 0.05) == pytest.approx(60 / 0.07)
    assert va.residual_income_value(500, 20, 0.1, 0.04) == pytest.approx(500 + 20 / 0.06)
    assert va.enterprise_value(900, 200, 50) == 1050
    assert va.enterprise_value(-50, -100, 30) == -180
    assert va.roic(120, 1000) == pytest.approx(0.12)
    assert va.eva(120, 1000, 0.08) == pytest.approx(40)
    assert va.ddm_value(0, 0.1, 0) == 0
    assert va.ddm_value(-2, 0, -0.1) == -20
    assert va.fcff_value(0, 0.1, 0) == 0
    assert va.fcff_value(-100, -0.05, -0.1) == -2000
    assert va.fcfe_value(0, 0.1, 0) == 0
    assert va.fcfe_value(-60, 0, -0.1) == -600
    assert va.residual_income_value(-50, 0, 0.1, 0) == -50
    assert va.residual_income_value(50, -20, 0, -0.1) == -150
    assert va.enterprise_value(0, 0, 0) == 0
    assert va.roic(0, 1000) == 0
    assert va.roic(120, -1000) == pytest.approx(-0.12)
    assert va.eva(120, 0, 0.08) == 120
    assert va.eva(120, -1000, 0.08) == 200
    assert va.eva(120, 1000, 0) == 120


def test_valuation_growth_and_zero_capital_guards() -> None:
    with pytest.raises(ValueError, match=re.escape("required return must exceed growth")):
        va.ddm_value(2, 0.04, 0.04)
    with pytest.raises(ValueError, match=re.escape("wacc must exceed growth")):
        va.fcff_value(100, 0.04, 0.05)
    with pytest.raises(ValueError, match=re.escape("cost of equity must exceed growth")):
        va.fcfe_value(60, 0.05, 0.05)
    with pytest.raises(ValueError, match=re.escape("required return must exceed growth")):
        va.residual_income_value(500, 20, 0.03, 0.04)
    with pytest.raises(ValueError, match=re.escape("zero invested capital")):
        va.roic(120, 0)
    with pytest.raises(ValueError):
        va.ddm_value(2, 0.03, 0.04)
    with pytest.raises(ValueError):
        va.fcff_value(100, 0.04, 0.04)
    with pytest.raises(ValueError):
        va.fcfe_value(60, 0.04, 0.05)
    with pytest.raises(ValueError):
        va.residual_income_value(500, 20, 0.04, 0.04)


def test_execution_cost_and_borrow_known_answers() -> None:
    assert ex.almgren_chriss_cost(1000, 5, 0.01, 0.02) == pytest.approx(5000 + 4000)
    assert ex.almgren_chriss_cost(-1000, 5, 0.01, 0.02) == pytest.approx(5000 + 4000)
    assert ex.days_to_cover(1_000_000, 250_000) == pytest.approx(4)
    assert ex.days_to_cover(0, 250_000) == 0
    assert ex.borrow_cost(1_000_000, 0.05, 30, 360) == pytest.approx(1_000_000 * 0.05 * 30 / 360)
    assert ex.almgren_chriss_cost(0, 5, 0.01, 0.02) == 0
    assert ex.almgren_chriss_cost(10, 2, 0, 0.4) == 20
    assert ex.almgren_chriss_cost(10, 2, 0.4, 0) == 20
    assert ex.days_to_cover(-100, 25) == -4
    assert ex.days_to_cover(100, -25) == -4
    assert ex.borrow_cost(1000, 0.06, -30, 360) == -5
    assert ex.borrow_cost(-1000, 0.06, 30, 360) == -5
    assert ex.borrow_cost(1000, -0.06, 30, 360) == -5
    assert ex.borrow_cost(1000, 0.06, 30, -360) == -5
    assert ex.borrow_cost(1000, 0.06, 0, 360) == 0
    assert ex.borrow_cost(1000, 0, 30, 360) == 0
    assert ex.borrow_cost(0, 0.06, 30, 360) == 0
    assert ex.almgren_chriss_cost(1000, 0.5, 0.01, 0.02) == pytest.approx(45000.0)


def test_black_scholes_maturity_two_absolute_greeks() -> None:
    call = ex.black_scholes_greeks(100, 100, 0.05, 0.2, 2.0, True)
    put = ex.black_scholes_greeks(100, 100, 0.05, 0.2, 2.0, False)
    assert call == pytest.approx(
        {
            "delta": 0.6896910267811551,
            "gamma": 0.012478546401807622,
            "vega": 49.91418560723049,
            "theta": -5.137825428018369,
        }
    )
    assert put == pytest.approx(
        {
            "delta": -0.3103089732188449,
            "gamma": 0.012478546401807622,
            "vega": 49.91418560723049,
            "theta": -0.6136383378385715,
        }
    )


def test_execution_cost_and_borrow_guards() -> None:
    with pytest.raises(ValueError, match=re.escape("nonpositive horizon")):
        ex.almgren_chriss_cost(1000, 0, 0.01, 0.02)
    with pytest.raises(ValueError):
        ex.almgren_chriss_cost(1000, -1, 0.01, 0.02)
    with pytest.raises(ValueError, match=re.escape("zero adv")):
        ex.days_to_cover(1000, 0)
    with pytest.raises(ValueError, match=re.escape("zero basis")):
        ex.borrow_cost(1_000_000, 0.05, 30, 0)


def test_fundamentals_leverage_known_answers() -> None:
    assert fu.net_debt_to_ebitda(500, 250) == pytest.approx(2)
    assert fu.net_debt_to_ebitda(-100, 250) == pytest.approx(-0.4)
    assert fu.interest_coverage(500, 100) == pytest.approx(5)
    assert fu.debt_to_equity(400, 800) == pytest.approx(0.5)
    assert fu.beneish_m_score(
        1, 2, 3, 4, 5, 6, 7, 8, -4.84, 0.92, 0.53, 0.40, 0.89, 0.11, -0.17, 0.32, -0.32
    ) == pytest.approx(1.11)
    assert fu.beneish_m_score(0, 0, 0, 0, 0, 0, 0, 0, -4.84, 1, 2, 3, 4, 5, 6, 7, 8) == -4.84
    assert fu.beneish_m_score(1, 2, 3, 4, 5, 6, 7, 8, -4.84, 0, 0, 0, 0, 0, 0, 0, 0) == -4.84


def test_fundamentals_leverage_guards() -> None:
    with pytest.raises(ValueError, match=re.escape("zero ebitda")):
        fu.net_debt_to_ebitda(500, 0)
    with pytest.raises(ValueError, match=re.escape("zero interest expense")):
        fu.interest_coverage(500, 0)
    with pytest.raises(ValueError, match=re.escape("zero equity")):
        fu.debt_to_equity(400, 0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: pr.beta_adjusted_net([1], []), "length mismatch"),
        (lambda: pr.unconstrained_mv_weights([1], [[1, 2]], 1), "length mismatch"),
        (lambda: pr.unconstrained_mv_weights([1], [[1]], 0), "nonpositive risk aversion"),
        (lambda: pr.unconstrained_mv_weights([1], [[1]], -1), "nonpositive risk aversion"),
        (lambda: pr.portfolio_variance([1], []), "length mismatch"),
        (lambda: pr.marginal_risk_contribution([1], [[0]]), "zero variance"),
        (lambda: pr.marginal_var([1], [[0]], 1), "zero variance"),
        (lambda: pr.systematic_variance([1], [], [[1]]), "length mismatch"),
        (lambda: pr.systematic_variance([1], [[1]], [[1, 2]]), "shape mismatch"),
        (lambda: pr.systematic_variance([1], [[1, 2]], [[1]]), "length mismatch"),
        (lambda: pr.garch_update(-1, 0, 0, 0, 0), "negative omega"),
        (lambda: pr.garch_update(0, -1, 0, 0, 0), "negative alpha"),
        (lambda: pr.garch_update(0, 0, -1, 0, 0), "negative beta"),
        (lambda: pr.garch_update(0, 0, 0, -1, 0), "negative prev_var"),
        (lambda: pr.volatility_target_scale(0, 1, 1), "nonpositive target vol"),
        (lambda: pr.volatility_target_scale(1, 0, 1), "nonpositive forecast vol"),
        (lambda: pr.volatility_target_scale(1, 1, 0), "nonpositive cap"),
        (lambda: pr.volatility_target_scale(-1, 1, 1), "nonpositive target vol"),
        (lambda: pr.volatility_target_scale(1, -1, 1), "nonpositive forecast vol"),
        (lambda: pr.volatility_target_scale(1, 1, -1), "nonpositive cap"),
        (lambda: pr.equal_weights(0), "nonpositive count"),
        (lambda: pr.cap_weights([], 0), "nonpositive cap"),
        (lambda: pr.cap_weights([], 1), "empty series"),
        (lambda: pr.cap_weights([-1], 1), "negative weight"),
        (lambda: pr.cap_weights([1, 1], 0.5), "infeasible cap for budget"),
        (lambda: pr.score_weights([1, -1]), "zero total"),
        (lambda: pr.inverse_vol_weights([]), "empty series"),
        (lambda: pr.inverse_vol_weights([1, 0]), "nonpositive vol"),
        (lambda: pr.inverse_vol_weights([1, -1]), "nonpositive vol"),
        (lambda: pr.diversification_ratio([1], [], [[1]]), "length mismatch"),
        (lambda: pr.diversification_ratio([1], [1], [[0]]), "zero variance"),
        (lambda: pr.tracking_error([1], [], 1), "length mismatch"),
        (lambda: pr.tracking_error([], [], 1), "empty series"),
        (lambda: pr.active_weights([1], []), "length mismatch"),
        (lambda: pr.active_weights([], []), "empty series"),
        (lambda: pr.portfolio_turnover([1], []), "length mismatch"),
        (lambda: pr.portfolio_turnover([], []), "empty series"),
        (lambda: pr.kelly_binary(-0.1, 1, 1), "prob out of range"),
        (lambda: pr.kelly_binary(1.1, 1, 1), "prob out of range"),
        (lambda: pr.kelly_binary(0.5, 0, 1), "nonpositive win"),
        (lambda: pr.kelly_binary(0.5, 1, 0), "nonpositive loss"),
        (lambda: pr.kelly_objective(-1, 0.5, 1, 1), "nonpositive 1 + fraction * win"),
        (lambda: pr.kelly_objective(1, 0.5, 1, 1), "nonpositive 1 - fraction * loss"),
        (lambda: pr.kelly_objective(-2, 0.5, 1, 1), "nonpositive 1 + fraction * win"),
        (lambda: pr.kelly_objective(2, 0.5, 1, 1), "nonpositive 1 - fraction * loss"),
        (lambda: pr.sample_covariance([1], [], 0), "length mismatch"),
        (lambda: pr.sample_covariance([1], [1], 1), "nonpositive degrees of freedom"),
        (lambda: pr.sample_correlation([1], []), "length mismatch"),
        (lambda: pr.sample_correlation([], []), "empty series"),
        (lambda: pr.sample_correlation([1, 1], [1, 2]), "zero variance"),
        (lambda: pr.ewma_cov_update([], [1], 0.5), "empty series"),
        (lambda: pr.ewma_cov_update([[1]], [], 0.5), "empty series"),
        (lambda: pr.ewma_cov_update([[1]], [1], -0.1), "decay out of range"),
        (lambda: pr.ewma_cov_update([[1]], [1], 1.1), "decay out of range"),
        (lambda: pr.ewma_cov_update([[1]], [1, 2], 0.5), "shape mismatch"),
        (lambda: pr.ewma_cov_update([[1, 2]], [1], 0.5), "shape mismatch"),
        (lambda: pr.realized_volatility_hf([], 1), "empty series"),
        (lambda: pr.shrinkage_covariance([[1]], [[1]], -0.1), "delta out of range"),
        (lambda: pr.shrinkage_covariance([[1]], [[1]], 1.1), "delta out of range"),
        (lambda: pr.shrinkage_covariance([], [[1]], 0.5), "shape mismatch"),
        (lambda: pr.shrinkage_covariance([[1]], [[1, 2]], 0.5), "shape mismatch"),
        (lambda: pr.skewness([]), "empty series"),
        (lambda: pr.skewness([1, 1]), "zero variance"),
        (lambda: pr.excess_kurtosis([]), "empty series"),
        (lambda: pr.excess_kurtosis([1, 1]), "zero variance"),
        (lambda: pr.active_share([1], []), "length mismatch"),
        (lambda: pr.active_share([], []), "empty series"),
        (lambda: pr.value_at_risk_historical([], 0.5), "empty series"),
        (lambda: pr.value_at_risk_historical([1], 0), "confidence not in (0, 1)"),
        (lambda: pr.value_at_risk_historical([1], 1), "confidence not in (0, 1)"),
        (lambda: pr.value_at_risk_parametric([], 1), "empty series"),
        (lambda: pr.expected_shortfall([], 0.5), "empty series"),
        (lambda: pr.expected_shortfall([1], 0), "confidence not in (0, 1)"),
        (lambda: pr.expected_shortfall([1], 1), "confidence not in (0, 1)"),
        (lambda: pr.max_drawdown([]), "empty series"),
        (lambda: pr.time_under_water([]), "empty series"),
        (lambda: pr.tail_risk_measure([1], []), "length mismatch"),
        (lambda: pr.tail_risk_measure([], []), "empty series"),
    ],
)
def test_portfolio_errors(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        call()
    assert str(error.value) == message


def test_technical_returns_averages_and_trends() -> None:
    assert te.log_return(110, 100) == pytest.approx(math.log(1.1))
    assert te.cumulative_return([0.1, -0.1]) == pytest.approx(-0.01)
    assert te.cumulative_return([0.1, -1, 0.5]) == -1
    assert te.annualized_return(0.1, 6, 12) == pytest.approx(0.21)
    assert te.sma([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
    assert te.sma([1, 2, 3, 4], 4) == [2.5]
    assert te.sma([1, 2], 1) == [1, 2]
    assert te.ema([1, 3, 5], 0.5) == [1, 2, 3.5]
    assert te.ema([1, 3, 5], 1) == [1, 3, 5]
    assert te.macd([1, 3], 1, 3, 3) == {"macd_line": 1, "signal_line": 0.5, "histogram": 0.5}
    assert te.macd([4, 4, 4], 1, 3, 2) == {"macd_line": 0, "signal_line": 0, "histogram": 0}
    assert te.rate_of_change([100, 105, 110], 2, 100) == pytest.approx(10)
    assert te.average_true_range([1, 2, 5], 2) == 3.5
    assert te.bollinger_bands([1, 2, 3, 4], 4, 2) == pytest.approx(
        {"middle": 2.5, "upper": 2.5 + math.sqrt(5), "lower": 2.5 - math.sqrt(5)}
    )
    assert te.bollinger_bands([1, 2, 3, 4], 2, 0) == {"middle": 3.5, "upper": 3.5, "lower": 3.5}
    assert te.high_52w_proximity([100, 120, 90], 2, 100) == 75
    assert te.high_52w_proximity([100, 120, 90], 1, 100) == 100


def test_technical_volatility_and_volume() -> None:
    assert te.parkinson_variance([math.e, math.e], [1, 1]) == pytest.approx(1 / (4 * math.log(2)))
    assert te.garman_klass_variance([1], [math.e], [1], [1]) == 0.5
    assert te.garman_klass_variance([1], [1], [1], [math.e]) == 0
    assert te.on_balance_volume([10, 11, 10, 10], [100, 200, 50, 999]) == [100, 300, 250, 250]
    assert te.accumulation_distribution([12, 10, 10], [8, 10, 6], [11, 10, 6], [100, 999, 50]) == [50, 50, 0]
    assert te.advance_decline_line([10, 5, 7], [3, 8, 7]) == [7, 4, 4]
    assert te.rsi([1, 3], [1, 1], 2, 100) == pytest.approx(100 * 2 / 3)
    assert te.rsi([0, 0], [0, 0], 2, 100) == 50
    assert te.rsi([1, 2], [0, 0], 2, 100) == 100
    assert te.rsi([0, 0], [1, 2], 2, 100) == 0
    typical = [10.0, 11.0, 10.0, 12.0]
    assert te.money_flow_index(typical, typical, typical, [1, 1, 1, 1], 3, 100) == pytest.approx(100 * 23 / 33)
    assert te.money_flow_index([10, 10], [10, 10], [10, 10], [1, 1], 1, 100) == 100
    assert te.money_flow_index([10, 9], [10, 9], [10, 9], [1, 1], 1, 100) == 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: te.log_return(0, 1), "nonpositive price"),
        (lambda: te.log_return(1, 0), "nonpositive price"),
        (lambda: te.cumulative_return([]), "empty series"),
        (lambda: te.annualized_return(0, 0, 1), "nonpositive num_periods"),
        (lambda: te.annualized_return(0, 1, 0), "nonpositive periods_per_year"),
        (lambda: te.annualized_return(-1, 1, 1), "total_return below -100%"),
        (lambda: te.sma([], 1), "empty series"),
        (lambda: te.sma([1], 0), "bad window"),
        (lambda: te.sma([1], 2), "bad window"),
        (lambda: te.ema([], 0.5), "empty series"),
        (lambda: te.ema([1], 0), "bad alpha"),
        (lambda: te.ema([1], 1.1), "bad alpha"),
        (lambda: te.macd([], 1, 2, 1), "empty series"),
        (lambda: te.macd([1], 0, 2, 1), "bad window"),
        (lambda: te.macd([1], 1, 0, 1), "bad window"),
        (lambda: te.macd([1], 1, 2, 0), "bad window"),
        (lambda: te.rate_of_change([], 1, 1), "empty series"),
        (lambda: te.rate_of_change([1, 2], 0, 1), "bad lookback"),
        (lambda: te.rate_of_change([1, 2], 2, 1), "bad lookback"),
        (lambda: te.rate_of_change([0, 2], 1, 1), "zero price_ago"),
        (lambda: te.rsi([1], [], 1, 100), "length mismatch"),
        (lambda: te.rsi([], [], 1, 100), "empty series"),
        (lambda: te.rsi([1], [1], 0, 100), "bad period"),
        (lambda: te.rsi([1], [1], 2, 100), "bad period"),
        (lambda: te.bollinger_bands([], 1, 1), "empty series"),
        (lambda: te.bollinger_bands([1], 0, 1), "bad window"),
        (lambda: te.bollinger_bands([1], 2, 1), "bad window"),
        (lambda: te.bollinger_bands([1], 1, -1), "negative num_std"),
        (lambda: te.average_true_range([], 1), "empty series"),
        (lambda: te.average_true_range([1], 0), "bad period"),
        (lambda: te.average_true_range([1], 2), "bad period"),
        (lambda: te.parkinson_variance([1], []), "length mismatch"),
        (lambda: te.parkinson_variance([], []), "empty series"),
        (lambda: te.parkinson_variance([0], [1]), "nonpositive price"),
        (lambda: te.parkinson_variance([1], [0]), "nonpositive price"),
        (lambda: te.garman_klass_variance([1], [1], [1], []), "length mismatch"),
        (lambda: te.garman_klass_variance([], [], [], []), "empty series"),
        (lambda: te.garman_klass_variance([0], [1], [1], [1]), "nonpositive price"),
        (lambda: te.garman_klass_variance([1], [0], [1], [1]), "nonpositive price"),
        (lambda: te.garman_klass_variance([1], [1], [0], [1]), "nonpositive price"),
        (lambda: te.garman_klass_variance([1], [1], [1], [0]), "nonpositive price"),
        (lambda: te.on_balance_volume([1], []), "length mismatch"),
        (lambda: te.on_balance_volume([], []), "empty series"),
        (lambda: te.money_flow_index([1], [1], [1], [], 1, 100), "length mismatch"),
        (lambda: te.money_flow_index([], [], [], [], 1, 100), "empty series"),
        (lambda: te.money_flow_index([1, 2], [1, 2], [1, 2], [1, 1], 0, 100), "bad period"),
        (lambda: te.money_flow_index([1, 2], [1, 2], [1, 2], [1, 1], 2, 100), "bad period"),
        (lambda: te.accumulation_distribution([1], [1], [1], []), "length mismatch"),
        (lambda: te.accumulation_distribution([], [], [], []), "empty series"),
        (lambda: te.high_52w_proximity([], 1, 100), "empty series"),
        (lambda: te.high_52w_proximity([1], 0, 100), "bad lookback"),
        (lambda: te.high_52w_proximity([1], 2, 100), "bad lookback"),
        (lambda: te.high_52w_proximity([0], 1, 100), "zero high"),
        (lambda: te.advance_decline_line([1], []), "length mismatch"),
        (lambda: te.advance_decline_line([], []), "empty series"),
    ],
)
def test_technical_errors(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        call()
    assert str(error.value) == message


def test_nonfinite_arithmetic_keeps_existing_semantics() -> None:
    nan = float("nan")
    inf = float("inf")
    assert ex.vwap([inf], [1]) == inf
    assert math.isnan(ex.pair_distance([inf], [inf], 1))
    assert math.isnan(fa.beta([nan, 1], [1, 2]))
    assert math.isnan(fa.size_log(nan, 10))
    assert math.isnan(fa.ml_feature_row({"a": nan}, {"a": 0}, {"a": 1}, {"a": 0}, {"a": 2})["a"])
    assert math.isnan(pe.time_weighted_return([nan]))
    assert math.isnan(pr.sample_covariance([nan, 1], [1, 2], 0))
    assert all(math.isnan(w) for w in pr.cap_weights([nan, 1], 1))
    assert all(math.isnan(w) for w in pr.cap_weights([1, inf], inf))
    assert pr.cap_weights([0.2, 0.3], nan) == pytest.approx([0.2, 0.3])
    assert pr.cap_weights([0.2, 0.3], inf) == pytest.approx([0.2, 0.3])
    assert math.isnan(te.log_return(nan, 1))
    assert math.isnan(te.parkinson_variance([nan], [1]))
    assert math.isnan(fu.dso(nan, 1, 360))
    assert math.isnan(va.oj_value(nan, 1, 1, 0.1, 0))
    with pytest.raises(ValueError, match="probability outside unit interval"):
        pe.brier_score([nan], [0])
    with pytest.raises(ValueError, match="negative weight"):
        pr.cap_weights([-inf], 1)


@seed(20261014)
@settings(max_examples=150, derandomize=True)
@given(
    st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=12),
    st.floats(min_value=1.000001, max_value=3, allow_nan=False, allow_infinity=False),
)
def test_prop_capped_allocations_preserve_budget_and_order(raw: list[int], headroom: float) -> None:
    weights = [value / 10 for value in raw]
    budget = sum(weights)
    cap = max(budget / len(weights) * headroom, 0.01)
    result = pr.cap_weights(weights, cap)
    assert sum(result) == pytest.approx(budget, rel=1e-8, abs=1e-8)
    assert all(0 <= weight <= cap for weight in result)
    assert pr.cap_weights(result, cap) == pytest.approx(result)
    for i in range(len(weights)):
        for j in range(len(weights)):
            if weights[i] < weights[j]:
                assert result[i] <= result[j] + 1e-9
            if weights[i] > 0 and weights[j] > 0 and result[i] < cap and result[j] < cap:
                assert result[i] / result[j] == pytest.approx(weights[i] / weights[j])


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(min_value=-1000, max_value=1000), min_size=2, max_size=20, unique=True),
    st.sampled_from([-10, -7, -3, -2, -1, 1, 2, 3, 7, 10]),
    st.integers(min_value=-100, max_value=100),
)
def test_prop_linear_exposure_and_correlation(market: list[int], slope: int, intercept: int) -> None:
    xs = [float(value) for value in market]
    ys = [slope * value + intercept for value in xs]
    assert fa.beta(ys, xs) == pytest.approx(slope)
    expected_correlation = math.copysign(1, slope)
    assert pr.sample_correlation(xs, ys) == pytest.approx(expected_correlation)
    assert ex.information_coefficient(xs, ys) == pytest.approx(expected_correlation)
    assert pr.sample_covariance(xs, ys, 1) == pytest.approx(pr.sample_covariance(ys, xs, 1))


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=20),
    st.integers(min_value=-50, max_value=50),
    st.integers(min_value=0, max_value=100),
)
def test_prop_scaling_clipping_and_drawdown(raw: list[int], lower: int, width: int) -> None:
    values = [float(value) for value in raw]
    weights = pr.inverse_vol_weights(values)
    assert sum(weights) == pytest.approx(1)
    assert all(weight > 0 for weight in weights)
    assert pr.inverse_vol_weights([value * 7 for value in values]) == pytest.approx(weights)
    clipped = ex.winsorize(values, lower, lower + width)
    assert all(lower <= value <= lower + width for value in clipped)
    assert ex.winsorize(clipped, lower, lower + width) == clipped
    assert 0 <= pr.max_drawdown(values) < 1
    assert pr.max_drawdown([value * 7 for value in values]) == pytest.approx(pr.max_drawdown(values))


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.floats(min_value=10, max_value=500, allow_nan=False, allow_infinity=False),
    st.floats(min_value=10, max_value=500, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.05, max_value=0.2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.05, max_value=1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
)
def test_prop_option_greeks_obey_put_call_parity(
    spot: float, strike: float, rate: float, sigma: float, maturity: float
) -> None:
    call = ex.black_scholes_greeks(spot, strike, rate, sigma, maturity, True)
    put = ex.black_scholes_greeks(spot, strike, rate, sigma, maturity, False)
    assert call["delta"] - put["delta"] == pytest.approx(1)
    assert call["gamma"] == put["gamma"]
    assert call["vega"] == put["vega"]
    assert call["theta"] - put["theta"] == pytest.approx(-rate * strike * math.exp(-rate * maturity), abs=1e-10)
    scaled = ex.black_scholes_greeks(2 * spot, 2 * strike, rate, sigma, maturity, True)
    assert scaled == pytest.approx(
        {"delta": call["delta"], "gamma": call["gamma"] / 2, "vega": 2 * call["vega"], "theta": 2 * call["theta"]}
    )


@pytest.mark.parametrize("ratio", [fu.net_debt_to_ebitda, fu.interest_coverage, fu.debt_to_equity])
def test_fundamental_ratios_keep_zero_and_negative_signs(ratio: Callable[[float, float], float]) -> None:
    assert ratio(0, 5) == 0
    assert ratio(-15, 5) == -3
    assert ratio(15, -5) == -3
    assert ratio(-15, -5) == 3


@pytest.mark.parametrize(
    "call",
    [
        lambda: pe.sharpe_ratio([-1, 1], 0, -1),
        lambda: pe.sortino_ratio([-1, 1], 0, -1),
        lambda: ex.black_scholes_greeks(-1, 1, 0, 1, 1, True),
        lambda: ex.black_scholes_greeks(1, -1, 0, 1, 1, True),
        lambda: ex.black_scholes_greeks(1, 1, 0, -1, 1, True),
        lambda: ex.black_scholes_greeks(1, 1, 0, 1, -1, True),
        lambda: ex.implementation_shortfall(1, 1, 1, 0, -1, 1, 1),
    ],
)
def test_calculation_negative_input_guards(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-100, 100), min_size=1, max_size=20),
    st.integers(1, 9),
    st.integers(-20, 20),
    st.integers(1, 10),
)
def test_prop_tail_risk_preserves_order_shift_and_scale(
    returns: list[int], confidence_tenths: int, shift: int, scale: int
) -> None:
    values = [value / 100 for value in returns]
    confidence = confidence_tenths / 10
    historical = pr.value_at_risk_historical(values, confidence)
    shortfall = pr.expected_shortfall(values, confidence)
    assert shortfall >= historical - 1e-12
    for measure in (pr.value_at_risk_historical, pr.expected_shortfall):
        baseline = measure(values, confidence)
        assert measure(list(reversed(values)), confidence) == pytest.approx(baseline)
        assert measure([scale * value + shift for value in values], confidence) == pytest.approx(
            scale * baseline - shift
        )
    parametric = pr.value_at_risk_parametric(values, 1.5)
    assert pr.value_at_risk_parametric([scale * value + shift for value in values], 1.5) == pytest.approx(
        scale * parametric - shift
    )


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.floats(0.1, 0.9, allow_nan=False, allow_infinity=False),
    st.floats(0.25, 3, allow_nan=False, allow_infinity=False),
    st.floats(0.25, 3, allow_nan=False, allow_infinity=False),
)
def test_prop_kelly_fraction_maximizes_admissible_log_growth(prob: float, win: float, loss: float) -> None:
    fraction = pr.kelly_binary(prob, win, loss)
    step = min(fraction + 1 / win, 1 / loss - fraction) / 4
    optimum = pr.kelly_objective(fraction, prob, win, loss)
    assert optimum >= -1e-14
    assert optimum > pr.kelly_objective(fraction - step, prob, win, loss)
    assert optimum > pr.kelly_objective(fraction + step, prob, win, loss)


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-100, 100), min_size=2, max_size=2),
    st.integers(1, 10),
    st.integers(1, 20),
    st.integers(1, 20),
    st.integers(1, 10),
)
def test_prop_risk_sizing_and_signed_exposures(
    returns: list[int], multiplier: int, target: int, forecast: int, cap: int
) -> None:
    values = [float(value) for value in returns]
    weights = pr.unconstrained_mv_weights(values, [[2, -1], [-1, 3]], 2)
    assert pr.unconstrained_mv_weights([multiplier * value for value in values], [[2, -1], [-1, 3]], 2) == (
        pytest.approx([multiplier * weight for weight in weights])
    )
    assert pr.unconstrained_mv_weights(values, [[2, -1], [-1, 3]], 2 * multiplier) == pytest.approx(
        [weight / multiplier for weight in weights]
    )
    gross = pr.gross_exposure(weights)
    net = pr.net_exposure(weights)
    assert gross >= abs(net)
    assert pr.gross_exposure([-multiplier * weight for weight in weights]) == pytest.approx(multiplier * gross)
    assert pr.net_exposure([-multiplier * weight for weight in weights]) == pytest.approx(-multiplier * net)
    scale = pr.volatility_target_scale(target, forecast, cap)
    assert 0 < scale <= cap
    assert pr.volatility_target_scale(target * multiplier, forecast * multiplier, cap) == pytest.approx(scale)


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.integers(-1000, 1000),
    st.integers(-1000, 1000),
    st.integers(-10, 10),
    st.integers(1, 20),
    st.integers(1, 10),
)
def test_prop_valuation_preserves_units_and_discount_spreads(
    cash: int, book: int, growth_tenths: int, spread_tenths: int, scale: int
) -> None:
    growth = growth_tenths / 10
    rate = growth + spread_tenths / 10
    for model in (va.ddm_value, va.fcff_value, va.fcfe_value):
        value = model(cash, rate, growth)
        assert model(scale * cash, rate, growth) == pytest.approx(scale * value)
        assert model(cash, rate + 1, growth + 1) == pytest.approx(value)
        assert model(cash, rate + spread_tenths / 10, growth) == pytest.approx(value / 2)
    residual = va.residual_income_value(book, cash, rate, growth)
    assert va.residual_income_value(scale * book, scale * cash, rate, growth) == pytest.approx(scale * residual)
    assert va.residual_income_value(book + 7, cash, rate, growth) == pytest.approx(residual + 7)
    assert va.enterprise_value(book + cash, cash, cash) == book + cash
    assert va.enterprise_value(scale * book, scale * cash, 3 * scale) == scale * va.enterprise_value(book, cash, 3)
    capital = spread_tenths
    assert va.roic(scale * cash, scale * capital) == pytest.approx(va.roic(cash, capital))
    assert va.eva(cash, capital, growth) == pytest.approx(capital * (va.roic(cash, capital) - growth))
    assert va.eva(scale * cash, scale * capital, growth) == pytest.approx(scale * va.eva(cash, capital, growth))


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(st.integers(-1000, 1000), st.integers(1, 1000), st.integers(1, 10))
def test_prop_fundamental_ratios_preserve_units_and_signs(numerator: int, denominator: int, scale: int) -> None:
    for ratio in (fu.net_debt_to_ebitda, fu.interest_coverage, fu.debt_to_equity):
        value = ratio(numerator, denominator)
        assert value * denominator == pytest.approx(numerator)
        assert ratio(scale * numerator, scale * denominator) == pytest.approx(value)
        assert ratio(numerator, -denominator) == pytest.approx(-value)


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-20, 20), min_size=8, max_size=8),
    st.lists(st.integers(-20, 20), min_size=8, max_size=8),
    st.integers(-20, 20),
    st.integers(0, 7),
    st.integers(1, 10),
)
def test_prop_beneish_associates_each_component_with_its_weight(
    components: list[int], weights: list[int], intercept: int, index: int, increment: int
) -> None:
    values = dict(zip(("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"), components, strict=True))
    coefficients = dict(
        zip(("w_dsri", "w_gmi", "w_aqi", "w_sgi", "w_depi", "w_sgai", "w_lvgi", "w_tata"), weights, strict=True)
    )
    baseline = fu.beneish_m_score(**values, intercept=intercept, **coefficients)
    values[list(values)[index]] += increment
    assert fu.beneish_m_score(**values, intercept=intercept, **coefficients) == baseline + increment * weights[index]
    assert fu.beneish_m_score(**values, intercept=intercept + increment, **coefficients) == (
        baseline + increment * weights[index] + increment
    )


def test_valuation_pure_units_and_missing_inputs_fail_closed() -> None:
    assert va.pe(10.0, 2.0) == pytest.approx(5.0)
    assert va.earnings_yield(2.0, 10.0) == pytest.approx(0.2)
    assert va.dividend_yield(0.5, 10.0) == pytest.approx(0.05)
    assert va.payout_ratio(0.5, 2.0) == pytest.approx(0.25)
    for call in (
        lambda: va.pe(10.0, 0.0),
        lambda: va.earnings_yield(2.0, 0.0),
        lambda: va.payout_ratio(0.5, 0.0),
    ):
        with pytest.raises(ValueError, match="zero"):
            call()


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(st.integers(1, 100), st.integers(0, 100), st.integers(1, 10), st.integers(-100, 100))
def test_prop_execution_costs_preserve_size_and_time_units(
    quantity: int, filled_percent: int, scale: int, days: int
) -> None:
    filled = quantity * filled_percent / 100
    shortfall = ex.implementation_shortfall(100, 102, 105, filled, quantity, 1, 1)
    assert shortfall == pytest.approx((filled_percent * 0.02 + (100 - filled_percent) * 0.05) / 100)
    assert ex.implementation_shortfall(100, 102, 105, scale * filled, scale * quantity, 1, 1) == pytest.approx(
        shortfall
    )
    assert ex.implementation_shortfall(100 * scale, 102 * scale, 105 * scale, filled, quantity, scale, 1) == (
        pytest.approx(scale * shortfall)
    )
    cost = ex.almgren_chriss_cost(quantity, 5, 0.01, 0.02)
    assert ex.almgren_chriss_cost(scale * quantity, 5, 0.01, 0.02) == pytest.approx(scale * scale * cost)
    assert ex.almgren_chriss_cost(quantity, 10, 0, 0.02) == pytest.approx(
        ex.almgren_chriss_cost(quantity, 5, 0, 0.02) / 2
    )
    assert ex.days_to_cover(scale * quantity, scale * 100) == pytest.approx(ex.days_to_cover(quantity, 100))
    assert ex.borrow_cost(quantity, 0.05, days, 360) + ex.borrow_cost(quantity, 0.05, 7, 360) == pytest.approx(
        ex.borrow_cost(quantity, 0.05, days + 7, 360)
    )


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-100, 100), min_size=2, max_size=20, unique=True),
    st.integers(-20, 20),
    st.integers(1, 10),
)
def test_prop_performance_ratios_preserve_excess_returns(raw: list[int], shift: int, scale: int) -> None:
    values = [value / 100 for value in raw]
    target = max(values) + 1
    for ratio in (pe.sharpe_ratio, pe.sortino_ratio):
        baseline = ratio(values, target, 4)
        assert ratio([scale * value + shift for value in values], scale * target + shift, 4) == pytest.approx(baseline)
        assert ratio(values, target, 4 * scale * scale) == pytest.approx(scale * baseline)


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-90, 100), min_size=2, max_size=12),
    st.integers(1, 10),
)
def test_prop_time_weighted_return_composes_periods(raw: list[int], split: int) -> None:
    values = [value / 100 for value in raw]
    boundary = min(split, len(values) - 1)
    first = pe.time_weighted_return(values[:boundary])
    second = pe.time_weighted_return(values[boundary:])
    total = pe.time_weighted_return(values)
    assert total == pytest.approx((1 + first) * (1 + second) - 1)
    assert pe.time_weighted_return([*values, 0]) == total
    assert pe.time_weighted_return(list(reversed(values))) == pytest.approx(total)


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(st.integers(-20, 20), st.integers(1, 500), st.integers(1, 20), st.integers(0, 30))
def test_prop_deflated_sharpe_is_an_ordered_probability(
    observed_hundredths: int, observations: int, trials: int, std_hundredths: int
) -> None:
    observed = observed_hundredths / 100
    baseline = pe.deflated_sharpe(observed, observations, trials, std_hundredths / 100, 0, 3)
    higher = pe.deflated_sharpe(observed + 0.1, observations, trials, std_hundredths / 100, 0, 3)
    assert 0 <= baseline <= higher <= 1
    assert pe.deflated_sharpe(observed, observations, 1, std_hundredths / 100, 0, 3) == pytest.approx(
        NormalDist().cdf(observed * math.sqrt(observations))
    )


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(st.integers(1, 20), st.integers(1, 20), st.integers(1, 10), st.integers(0, 40))
def test_prop_walk_forward_splits_are_complete_chronological_windows(
    train_length: int, test_length: int, step: int, extra: int
) -> None:
    observations = train_length + test_length + extra
    result = pe.walk_forward_split(observations, train_length, test_length, step)
    assert len(result["train"]) == len(result["test"]) == extra // step + 1
    for index, (train, test) in enumerate(zip(result["train"], result["test"], strict=True)):
        assert len(train) == train_length
        assert len(test) == test_length
        assert train + test == list(range(index * step, index * step + train_length + test_length))
        assert set(train).isdisjoint(test)
        assert test[-1] < observations


@seed(20261014)
@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.tuples(st.integers(-20, 20), st.integers(0, 10)), min_size=1, max_size=20),
    st.integers(-10, 10),
    st.integers(0, 10),
    st.integers(0, 10),
)
def test_prop_purged_splits_exclude_overlap_and_embargo(
    events: list[tuple[int, int]], start: int, duration: int, embargo: int
) -> None:
    starts = [float(time) for time, _ in events]
    ends = [float(time + length) for time, length in events]
    end = start + duration
    result = pe.purged_split(starts, ends, start, end, embargo)
    assert set(result["train"]).isdisjoint(result["test"])
    for index, (event_start, event_end) in enumerate(zip(starts, ends, strict=True)):
        overlaps = max(event_start, start) <= min(event_end, end)
        assert (index in result["test"]) == overlaps
        assert (index in result["train"]) == (event_end < start or event_start > end + embargo)
    wider = pe.purged_split(starts, ends, start, end, embargo + 1)
    assert wider["test"] == result["test"]
    assert set(wider["train"]) <= set(result["train"])
