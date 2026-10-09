"""Point-in-time ticker resolution tests for the portfolio sync service.

The resolution rules under test:

- a ticker alias resolves to the SEC entity and security IDs;
- lookups are case-insensitive (UPPER);
- an alias whose ``known_at`` is after ``as_of`` is invisible (no
  retroactive contamination);
- two active aliases pointing at different entities resolve as ambiguous;
- unknown tickers resolve to ``resolved=False`` and never crash;
- ``provider_instrument_id`` is accepted but does not change the result.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from app.data_sources import SourceGateway
from app.domain.market.identity import resolve_ticker_aliases
from app.domain.market.securities import TickerAlias
from app.services.portfolio_sync import resolve_security

AMD_ENTITY = "sec:cik:0000320193"
AMD_SECURITY = "sec:equity:0000320193"


_ALIASES: list[TickerAlias] = []


def _alias_row(
    *,
    entity_id: str = AMD_ENTITY,
    security_id: str | None = AMD_SECURITY,
    alias: str = "AMD",
    known_at: str = "2026-01-01T00:00:00Z",
    source: str = "sec",
    valid_from: str | None = "2026-01-01",
    valid_to: str | None = None,
) -> TickerAlias:
    return TickerAlias(
        alias_type="ticker",
        alias_value=alias,
        entity_id=entity_id,
        security_id=security_id,
        source=source,
        valid_from=valid_from,
        valid_to=valid_to,
        known_at=known_at,
        retrieved_at=known_at,
    )


@pytest.fixture(autouse=True)
def _stub_gateway(monkeypatch: pytest.MonkeyPatch):
    """Seed the live-alias double; PIT stays in the resolver, never here."""
    _ALIASES.clear()
    _ALIASES.append(_alias_row())

    def _candidates(self: object, ticker: str, as_of: datetime) -> list[TickerAlias]:
        del self, as_of
        want = ticker.strip().upper()
        return [alias for alias in _ALIASES if alias.alias_value == want]

    monkeypatch.setattr(SourceGateway, "ticker_candidates", _candidates)


def _seed_aliases(*rows: TickerAlias) -> None:
    _ALIASES.clear()
    _ALIASES.extend(rows)


def _seed_amd(known_at: str = "2026-01-01T00:00:00Z") -> None:
    _seed_aliases(_alias_row(known_at=known_at))


def test_ticker_resolves_to_entity_and_security_ids() -> None:
    _seed_amd()
    resolution = resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is True
    assert resolution.resolution_method == "entity_alias"
    assert resolution.ticker == "AMD"
    assert resolution.entity_id == AMD_ENTITY
    assert resolution.security_id == AMD_SECURITY


def test_lookup_is_case_insensitive() -> None:
    _seed_amd()
    resolution = resolve_security("amd", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is True
    assert resolution.entity_id == AMD_ENTITY
    assert resolution.security_id == AMD_SECURITY


def test_unknown_ticker_is_unresolved_without_crashing() -> None:
    _seed_amd()
    resolution = resolve_security("NOTREAL", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is False
    assert resolution.resolution_method == "unresolved"
    assert resolution.entity_id is None
    assert resolution.security_id is None
    assert resolution.ticker == "NOTREAL"


def test_alias_known_after_as_of_is_invisible() -> None:
    _seed_amd(known_at="2026-09-01T12:00:00Z")
    early = resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert early.resolved is False
    assert early.resolution_method == "unresolved"
    on_day = resolve_security("AMD", as_of=datetime(2026, 9, 1, 13, 0, tzinfo=UTC))
    assert on_day.resolved is True
    assert on_day.entity_id == AMD_ENTITY


def test_ambiguous_when_multiple_entities_knowable() -> None:
    _seed_aliases(
        _alias_row(known_at="2026-01-01T12:00:00Z"),
        _alias_row(
            entity_id="sec:cik:0000320194",
            security_id="sec:equity:0000320194",
            known_at="2026-06-01T12:00:00Z",
        ),
    )
    before_relabel = resolve_security("AMD", as_of=datetime(2026, 3, 1, 0, 0, tzinfo=UTC))
    assert before_relabel.resolved is True
    assert before_relabel.entity_id == AMD_ENTITY
    assert before_relabel.security_id == AMD_SECURITY
    after_relabel = resolve_security("AMD", as_of=datetime(2026, 8, 1, 0, 0, tzinfo=UTC))
    assert after_relabel.resolved is False
    assert after_relabel.resolution_method == "ambiguous"
    assert after_relabel.entity_id is None
    assert after_relabel.security_id is None


def test_provider_instrument_id_does_not_change_resolution() -> None:
    _seed_amd()
    plain = resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    with_instrument = resolve_security(
        "AMD",
        provider_instrument_id="instr-amd",
        as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )
    assert with_instrument == plain


def test_default_as_of_is_today_and_resolves() -> None:
    _seed_amd(known_at="2026-01-01T00:00:00Z")
    resolution = resolve_security("AMD")
    assert resolution.entity_id == AMD_ENTITY
    assert datetime.now(UTC).date() >= date(2026, 1, 1)


def test_same_instant_z_and_offset_visible() -> None:
    _seed_amd(known_at="2026-08-25T12:00:00Z")
    resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))


def test_record_one_microsecond_after_as_of_invisible() -> None:
    _seed_amd(known_at="2026-08-25T12:00:00.000001Z")
    resolution = resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is False
    assert resolution.resolution_method == "unresolved"


def test_non_utc_offset_compares_chronologically() -> None:
    _seed_amd(known_at="2026-08-25T13:00:00+01:00")
    resolution = resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
    assert resolution.resolved is True
    assert resolution.entity_id == AMD_ENTITY


def test_naive_as_of_rejected() -> None:
    _seed_amd()
    with pytest.raises(ValueError):
        resolve_security("AMD", as_of=datetime(2026, 8, 25, 12, 0))  # noqa: DTZ001 - naive input is the case under test


def test_date_as_of_rejected() -> None:
    _seed_amd()
    with pytest.raises(TypeError):
        _resolve: Callable[..., object] = resolve_security
        _resolve("AMD", as_of=date(2026, 8, 25))  # runtime TypeError guard: date is not a valid as_of


def test_newest_alias_wins_chronologically_not_lexically() -> None:
    _seed_aliases(
        _alias_row(known_at="2026-08-25T13:00:00+01:00", security_id=None),
        _alias_row(known_at="2026-08-25T12:30:00Z", security_id=AMD_SECURITY, source="control"),
    )
    resolution = resolve_security("AMD", as_of=datetime(2026, 8, 26, 0, 0, tzinfo=UTC))
    assert resolution.resolved is True
    assert resolution.security_id == AMD_SECURITY


def test_same_instant_conflicting_securities_are_ambiguous() -> None:
    _seed_aliases(
        _alias_row(known_at="2026-08-25T12:00:00Z", security_id=AMD_SECURITY),
        _alias_row(known_at="2026-08-25T12:00:00Z", security_id="sec:equity:0000999999", source="control"),
    )
    resolution = resolve_security("AMD", as_of=datetime(2026, 8, 26, 0, 0, tzinfo=UTC))
    assert resolution.resolved is False
    assert resolution.resolution_method == "ambiguous"
    assert resolution.entity_id == AMD_ENTITY
    assert resolution.security_id is None


def test_same_instant_null_and_explicit_security_not_conflict() -> None:
    _seed_aliases(
        _alias_row(known_at="2026-08-25T12:00:00Z", security_id=None),
        _alias_row(known_at="2026-08-25T12:00:00Z", security_id=AMD_SECURITY, source="control"),
    )
    resolution = resolve_security("AMD", as_of=datetime(2026, 8, 26, 0, 0, tzinfo=UTC))
    assert resolution.resolved is True
    assert resolution.security_id == AMD_SECURITY


# ---------------------------------------------------------------------------
# Pure resolver (no storage): resolve_ticker_aliases over constructed aliases
# ---------------------------------------------------------------------------

AS_OF = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


def _alias(
    known_at: str,
    *,
    entity_id: str = AMD_ENTITY,
    security_id: str | None = AMD_SECURITY,
    source: str = "sec",
    valid_from: str | None = None,
    valid_to: str | None = None,
    retrieved_at: str | None = None,
) -> TickerAlias:
    return TickerAlias(
        alias_type="ticker",
        alias_value="AMD",
        entity_id=entity_id,
        security_id=security_id,
        source=source,
        valid_from=valid_from,
        valid_to=valid_to,
        known_at=known_at,
        retrieved_at=retrieved_at or known_at,
    )


def test_pure_future_knowledge_is_invisible() -> None:
    resolution = resolve_ticker_aliases("AMD", [_alias("2026-08-26T00:00:00Z")], as_of=AS_OF)
    assert resolution.resolved is False
    assert resolution.resolution_method == "unresolved"


def test_pure_expired_alias_is_invisible() -> None:
    resolution = resolve_ticker_aliases(
        "AMD",
        [_alias("2026-08-01T00:00:00Z", valid_from="2026-01-01", valid_to="2026-08-24")],
        as_of=AS_OF,
    )
    assert resolution.resolved is False
    assert resolution.resolution_method == "unresolved"


def test_pure_date_only_valid_to_boundary_expired_on_the_25th() -> None:
    resolution = resolve_ticker_aliases(
        "AMD",
        [_alias("2026-08-01T00:00:00Z", valid_from="2026-01-01", valid_to="2026-08-25")],
        as_of=datetime(2026, 8, 25, 0, 0, tzinfo=UTC),
    )
    assert resolution.resolved is False
    assert resolution.resolution_method == "unresolved"


def test_pure_two_simultaneously_valid_entities_ambiguous() -> None:
    resolution = resolve_ticker_aliases(
        "AMD",
        [
            _alias("2026-01-01T00:00:00Z"),
            _alias(
                "2026-01-01T00:00:00Z",
                entity_id="sec:cik:0000320194",
                security_id="sec:equity:0000320194",
                source="control",
            ),
        ],
        as_of=AS_OF,
    )
    assert resolution.resolved is False
    assert resolution.resolution_method == "ambiguous"
    assert resolution.entity_id is None
    assert resolution.security_id is None


def test_pure_same_instant_conflicting_security_ids_ambiguous() -> None:
    resolution = resolve_ticker_aliases(
        "AMD",
        [
            _alias("2026-08-25T12:00:00Z", security_id=AMD_SECURITY),
            _alias("2026-08-25T12:00:00Z", security_id="sec:equity:0000999999", source="control"),
        ],
        as_of=AS_OF,
    )
    assert resolution.resolved is False
    assert resolution.resolution_method == "ambiguous"
    assert resolution.entity_id == AMD_ENTITY
    assert resolution.security_id is None


def test_pure_explicit_id_equal_to_derived_sec_cik_id_not_ambiguous() -> None:
    # The resolver derives the missing security id of a sec:cik entity to
    # sec:equity:<cik>, so both rows resolve to the same id — one distinct
    # value, not a conflict.
    resolution = resolve_ticker_aliases(
        "AMD",
        [
            _alias("2026-08-25T12:00:00Z", security_id=None),
            _alias("2026-08-25T12:00:00Z", security_id=AMD_SECURITY, source="control"),
        ],
        as_of=AS_OF,
    )
    assert resolution.resolved is True
    assert resolution.resolution_method == "entity_alias"
    assert resolution.security_id == AMD_SECURITY


def test_pure_older_mapping_then_newer_mapping_newer_wins() -> None:
    resolution = resolve_ticker_aliases(
        "AMD",
        [
            _alias("2026-08-25T12:00:00Z", security_id="sec:equity:0000999999"),
            _alias("2026-08-25T13:00:00Z", security_id=AMD_SECURITY, source="control"),
        ],
        as_of=datetime(2026, 8, 26, 0, 0, tzinfo=UTC),
    )
    assert resolution.resolved is True
    assert resolution.resolution_method == "entity_alias"
    assert resolution.security_id == AMD_SECURITY


def test_pure_nothing_usable_is_unresolved() -> None:
    resolution = resolve_ticker_aliases("AMD", [], as_of=AS_OF)
    assert resolution.resolved is False
    assert resolution.resolution_method == "unresolved"
    assert resolution.ticker == "AMD"


def test_pure_timezone_offset_known_at_chronological_not_lexical() -> None:
    resolution = resolve_ticker_aliases(
        "AMD",
        [
            # 13:00+01:00 == 12:00Z: sorts first lexically but is older.
            _alias("2026-08-25T13:00:00+01:00", security_id=None),
            _alias("2026-08-25T12:30:00Z", security_id=AMD_SECURITY, source="control"),
        ],
        as_of=datetime(2026, 8, 26, 0, 0, tzinfo=UTC),
    )
    assert resolution.resolved is True
    assert resolution.security_id == AMD_SECURITY


def test_pure_naive_as_of_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_ticker_aliases("AMD", [_alias("2026-01-01T00:00:00Z")], as_of=datetime(2026, 8, 25, 12, 0))  # noqa: DTZ001 - naive input is the case under test


def test_pure_date_as_of_rejected() -> None:
    with pytest.raises(TypeError):
        _aliases: Callable[..., object] = resolve_ticker_aliases
        _aliases(
            "AMD", [_alias("2026-01-01T00:00:00Z")], as_of=date(2026, 8, 25)
        )  # runtime TypeError guard: date is not a valid as_of


def test_pure_aware_non_utc_as_of_compares_chronologically() -> None:
    # 13:00+01:00 == 12:00Z: the alias is visible exactly at as_of.
    resolution = resolve_ticker_aliases(
        "AMD",
        [_alias("2026-08-25T12:00:00Z")],
        as_of=datetime(2026, 8, 25, 13, 0, tzinfo=timezone(timedelta(hours=1))),
    )
    assert resolution.resolved is True
    assert resolution.security_id == AMD_SECURITY
