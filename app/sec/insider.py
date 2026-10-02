"""Deterministic Forms 3/4/5 + 144 insider normalization (no network).

Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind these live readers, never inside normalization.
"""

from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path

from .cusip import normalize_cusip, normalize_isin
from .models import (
    Filing,
    InsiderTransaction,
    InstitutionalHolding,
    ProposedInsiderSale,
)

# `object` marks the edgar SDK dynamic boundary (no stubs): attrs are read
# via getattr and validated before building insider/13F domain objects.


_HOLDING_FORMS = ("13F-HR", "13F-HR/A")


TRANSACTION_KINDS = {
    "P": "open_market_purchase",
    "S": "open_market_sale",
    "M": "exercise",
    "X": "exercise",
    "A": "grant_award",
    "G": "gift",
    "C": "conversion",
    "F": "tax_withholding",
}

_DEFAULT_FORMS = ("3", "3/A", "4", "4/A", "5", "5/A")
_DEFAULT_144_FORMS = ("144", "144/A")


def list_sec_filings(
    ticker_or_cik: str | int,
    forms: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    as_of: str | date | datetime | None = None,
    limit: int | None = 50,
) -> list[Filing]:
    """Lazy seam: tests monkeypatch this name; real path imports on call."""
    from .filings import list_sec_filings as _real

    return _real(ticker_or_cik, forms=forms, start_date=start_date, end_date=end_date, as_of=as_of, limit=limit)


def classify_transaction(code: object) -> str:
    try:
        key = str(code).strip().upper()
    except Exception:  # noqa: BLE001 - untrusted code coerces to other, never raises
        return "other"
    return TRANSACTION_KINDS.get(key, "other")


def _safe_int(value: object) -> int | None:
    """Share counts truncate toward zero: SEC fractional shares (2605.75) must not null out."""
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, float):
            return int(value)
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in ("none", "nan", "na", "n/a", "--"):
            return None
        return int(float(text)) if "." in text else int(text)
    except ValueError, TypeError:
        return None


_TYPE_TO_ACQUIRED = {
    "purchase": "A",
    "sale": "D",
    "derivative_purchase": "A",
    "derivative_sale": "D",
    "exercise": "A",
    "award": "A",
    "tax": "D",
    "gift": "D",
    "conversion": "D",
    "other_acquisition": "A",
    "other_disposition": "D",
}


def _key_float(value: object) -> float | None:
    try:
        return float(str(value).strip().replace(",", "")) if value is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def _side_of_activity(activity: object) -> str:
    """Table side of one activity: derivative rows join derivative rows only."""
    if _str_or_none(_first(activity, "security_type")) == "derivative":
        return "derivative"
    if _first(activity, "is_derivative") is True:
        return "derivative"
    return "non-derivative"


def _enrichment_key_of(activity: object) -> tuple[str, str | None, float | None, float | None] | None:
    """Keyed join identity for one activity: (side, code, shares, price). Shared by both
    sides so a table reorder cannot misalign dates. None when unkeyable."""
    code = _str_or_none(_first(activity, "transaction_code", "code", "transaction_type"))
    shares = _key_float(_first(activity, "shares", "shares_numeric"))
    price = _key_float(_first(activity, "price_per_share", "price", "price_numeric"))
    if code is None and shares is None:
        return None
    return _side_of_activity(activity), code, shares, price


def _table_rows_of(obj: object) -> list[tuple[str, dict[str, object]]]:
    """Non-derivative + derivative table rows as (side, dict); [] when unavailable."""
    out: list[tuple[str, dict[str, object]]] = []
    for table_name, side in (("non_derivative_table", "non-derivative"), ("derivative_table", "derivative")):
        try:
            table = getattr(obj, table_name, None)
            data = getattr(getattr(table, "transactions", None), "data", None)
            rows = list(data.iterrows()) if data is not None else []
        except Exception:  # noqa: BLE001, S112 - table path degrades to [] on SDK shape change, never raises
            continue
        for _, row in rows:
            try:
                out.append((side, dict(row.to_dict()) if hasattr(row, "to_dict") else dict(row)))
            except Exception:  # noqa: BLE001, S112 - malformed row is skipped, the scan continues
                continue
    return out


def _enrichment_of(obj: object) -> dict[tuple[str, str | None, float | None, float | None], dict[str, object]]:
    """First-wins keyed index of table rows; duplicates refuse (no guessing)."""
    index: dict[tuple[str, str | None, float | None, float | None], dict[str, object]] = {}
    for side, row in _table_rows_of(obj):
        code = _str_or_none(row.get("Code"))
        key = (side, code, _key_float(row.get("Shares")), _key_float(row.get("Price")))
        if (code is None and key[2] is None) or key in index:
            continue
        index[key] = row
    return index


def _enriched_of(
    activity: object, index: dict[tuple[str, str | None, float | None, float | None], dict[str, object]]
) -> dict[str, object]:
    """Table row for one activity by key; {} on miss (never a wrong row)."""
    key = _enrichment_key_of(activity)
    if key is None:
        return {}
    return index.get(key) or {}


def _safe_float(value: object) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in ("none", "nan", "na", "n/a", "--"):
            return None
        return float(text)
    except ValueError, TypeError:
        return None


def _first(obj: object, *names: str) -> object:
    for name in names:
        try:
            value: object = getattr(obj, name)
        except Exception:  # noqa: BLE001, S112 - failed attr read tries the next name, never raises
            continue
        if value is not None:
            return value
    return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001 - untrusted value coerces to None, never raises
        return None
    return text or None


def _direct_cik_of(obj: object) -> str | None:
    cik = _first(obj, "insider_cik", "reporting_owner_cik", "owner_cik", "cik")
    if cik is None:
        return None
    try:
        return str(cik)
    except Exception:  # noqa: BLE001 - untrusted value coerces to None, never raises
        return None


def _owner_list_cik_of(obj: object) -> str | None:
    try:
        owners = getattr(obj, "reporting_owners", None) or []
        for owner in list(owners):
            cik = _first(owner, "cik", "owner_cik", "reporting_owner_cik")
            if cik is not None:
                return str(cik)
    except Exception:  # noqa: BLE001 - owner scan degrades to None on malformed payload, never raises
        return None
    return None


def _insider_cik(obj: object) -> str | None:
    return _direct_cik_of(obj) or _owner_list_cik_of(obj)


def _owner_candidates_of(obj: object) -> list[object]:
    try:
        container = getattr(obj, "reporting_owners", None)
        candidates = getattr(container, "owners", None) if container is not None else None
        if candidates is None and isinstance(container, (list, tuple)):
            candidates = container
        return list(candidates) if candidates else []
    except Exception:  # noqa: BLE001 - owner candidates degrade to empty on malformed payload, never raises
        return []


def _owner_record_of(owner: object) -> dict[str, object]:
    return {
        "name": _str_or_none(_first(owner, "name_unreversed", "name", "owner_name")),
        "cik": _str_or_none(_first(owner, "cik", "owner_cik", "reporting_owner_cik")),
        "is_director": _first(owner, "is_director"),
        "is_officer": _first(owner, "is_officer"),
        "is_ten_percent": _first(owner, "is_ten_pct_owner", "is_ten_percent"),
        "is_other": _first(owner, "is_other"),
        "role_title": _str_or_none(_first(owner, "officer_title", "role_title", "title")),
    }


def _fallback_owner_of(fallback_name: str | None, fallback_cik: str | None) -> list[dict[str, object]]:
    return [
        {
            "name": _str_or_none(fallback_name),
            "cik": _str_or_none(fallback_cik),
            "is_director": None,
            "is_officer": None,
            "is_ten_percent": None,
            "is_other": None,
            "role_title": None,
        }
    ]


def _owners_of(
    obj: object, fallback_name: str | None = None, fallback_cik: str | None = None
) -> list[dict[str, object]]:
    """Reporting owners: structured list first, legacy attrs as one owner."""
    out: list[dict[str, object]] = []
    for owner in _owner_candidates_of(obj):
        try:
            out.append(_owner_record_of(owner))
        except Exception:  # noqa: BLE001, S112 - malformed owner is skipped, the batch continues
            continue
    return out or _fallback_owner_of(fallback_name, fallback_cik)


def _issuer_of(obj: object) -> tuple[str | None, str | None, str | None]:
    """Authoritative issuer from structured ownership XML; never the filer."""
    try:
        issuer = getattr(obj, "issuer", None)
    except Exception:  # noqa: BLE001 - issuer read degrades to unknown on malformed payload, never raises
        return None, None, None
    if issuer is None or isinstance(issuer, str):
        return None, None, None
    cik = _str_or_none(_first(issuer, "cik", "issuer_cik", "issuerCIK"))
    name = _str_or_none(_first(issuer, "name", "issuer_name", "issuerName"))
    ticker = _str_or_none(_first(issuer, "ticker", "symbol"))
    return cik, name, ticker


def _bool_or_none(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().lower()
    except Exception:  # noqa: BLE001 - untrusted flag coerces to None, never raises
        return None
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return None


def _fallback_identity_of(obj: object) -> tuple[str | None, str | None] | None:
    try:
        return (_str_or_none(_first(obj, "insider_name", "reporting_owner", "owner_name", "name")), _insider_cik(obj))
    except Exception:  # noqa: BLE001 - fallback identity degrades to None on malformed payload, never raises
        return None


def _activity_rows_of(obj: object) -> list[object] | None:
    try:
        get_activities = getattr(obj, "get_transaction_activities")  # noqa: B009 - edgar SDK has no stubs; getattr keeps the dynamic boundary instead of failing the checker
        activities = get_activities()
    except Exception:  # noqa: BLE001 - activity fetch degrades to None on SDK failure, never raises
        return None
    if not activities:
        return None
    try:
        return list(activities)
    except Exception:  # noqa: BLE001 - activity list coercion degrades to None on SDK failure, never raises
        return None


def _resolved_issuer_of(obj: object, issuer: str, issuer_cik: str | int | None) -> tuple[str, str | None]:
    info_cik, info_name, _ticker = _issuer_of(obj)
    try:
        explicit_cik = str(issuer_cik).strip() if issuer_cik is not None else None
    except Exception:  # noqa: BLE001 - explicit CIK coerces to None, never raises
        explicit_cik = None
    return info_name or issuer, explicit_cik or info_cik


def _known_at_of(known_at: str | None, filed_at: str | None) -> str | None:
    try:
        return known_at if isinstance(known_at, str) else str(known_at) if known_at is not None else filed_at
    except Exception:  # noqa: BLE001 - known_at coerces to the filed_at fallback, never raises
        return filed_at


def _fallback_date_of(obj: object) -> str | None:
    """Filing-level earliest-transaction date; None when absent."""
    return _str_or_none(getattr(obj, "reporting_period", None))


def _date_of(activity: object, enriched: dict[str, object], fallback: str | None) -> str | None:
    """Per-row table date by key; reporting_period fallback; never a wrong row's date."""
    return (
        _str_or_none(_first(activity, "transaction_date", "date", "execution_date"))
        or _str_or_none(enriched.get("Date"))
        or fallback
    )


def _acquired_of(activity: object, enriched: dict[str, object]) -> str | None:
    """A/D from the activity, the keyed table row, or the SDK transaction_type; never guessed."""
    direct = _str_or_none(
        _first(activity, "acquired_disposed", "acquired_disposed_code", "acquired_or_disposed", "action", "buy_or_sell")
    )
    if direct is not None:
        return direct
    table = _str_or_none(enriched.get("AcquiredDisposed"))
    if table is not None:
        return table
    return _TYPE_TO_ACQUIRED.get(str(_first(activity, "transaction_type") or "").strip())


def _holdings_of(activity: object, enriched: dict[str, object]) -> int | None:
    """Direct holdings win, table Remaining on miss; 0 stays 0 (never falsy-fallback)."""
    direct = _safe_int(
        _first(activity, "holdings_after", "shares_owned_after", "holdings", "balance_after", "shares_held")
    )
    return direct if direct is not None else _safe_int(enriched.get("Remaining"))


def _activity_record_of(
    activity: object,
    owner: dict[str, object],
    *,
    issuer: str,
    form: str,
    filed_at: str | None,
    accession_no: str,
    issuer_cik: str | None,
    document_name: str | None,
    known: str | None,
    enriched: dict[str, object] | None = None,
    fallback_date: str | None = None,
) -> InsiderTransaction:
    code = _str_or_none(_first(activity, "transaction_code", "code", "transaction_type"))
    row = enriched or {}
    return InsiderTransaction(
        insider_name=_str_or_none(owner["name"]),
        insider_cik=_str_or_none(owner["cik"]),
        issuer=issuer,
        form=form,
        filed_at=filed_at,
        accession_no=accession_no,
        transaction_date=_date_of(activity, row, fallback_date),
        security=_str_or_none(_first(activity, "security", "security_title", "title", "security_name")),
        transaction_code=code,
        transaction_kind=classify_transaction(code),
        shares=_safe_int(
            _first(activity, "shares", "share_count", "num_shares", "amount", "shares_transacted", "shares_numeric")
        ),
        price=_safe_float(_first(activity, "price", "price_per_share", "price_numeric", "execution_price")),
        acquired_disposed=_acquired_of(activity, row),
        holdings_after=_holdings_of(activity, row),
        issuer_cik=issuer_cik,
        is_director=_bool_or_none(owner["is_director"]),
        is_officer=_bool_or_none(owner["is_officer"]),
        is_ten_percent=_bool_or_none(owner["is_ten_percent"]),
        is_other=_bool_or_none(owner["is_other"]),
        role_title=_str_or_none(owner["role_title"]),
        document_name=document_name,
        known_at=known,
    )


def normalize_ownership_filing(
    obj: object,
    *,
    issuer: str,
    form: str,
    filed_at: str | None,
    accession_no: str,
    issuer_cik: str | int | None = None,
    document_name: str | None = None,
    known_at: str | None = None,
) -> list[InsiderTransaction]:
    """One InsiderTransaction per (owner, activity) row; never raises."""
    fallback = _fallback_identity_of(obj)
    if fallback is None:
        return []
    rows = _activity_rows_of(obj)
    if not rows:
        return []
    resolved_issuer, resolved_issuer_cik = _resolved_issuer_of(obj, issuer, issuer_cik)
    known = _known_at_of(known_at, filed_at)
    owners = _owners_of(obj, fallback[0], fallback[1])
    index = _enrichment_of(obj)
    fallback_date = _fallback_date_of(obj)
    out: list[InsiderTransaction] = []
    for activity in rows:
        enriched = _enriched_of(activity, index)
        for owner in owners:
            try:
                out.append(
                    _activity_record_of(
                        activity,
                        owner,
                        issuer=resolved_issuer,
                        form=form,
                        filed_at=filed_at,
                        accession_no=accession_no,
                        issuer_cik=resolved_issuer_cik,
                        document_name=document_name,
                        known=known,
                        enriched=enriched,
                        fallback_date=fallback_date,
                    )
                )
            except Exception:  # noqa: BLE001, S112 - malformed activity is skipped, the filing continues
                continue
    return out


def load_ownership(accession_no: str) -> object:
    """Live seam: filing.obj() stays here; raises on failure."""
    from .documents import get_by_accession_number

    return get_by_accession_number(accession_no).obj()


def _issuer_name_of(obj: object, fallback: str) -> str:
    try:
        _cik, _name, _t = _issuer_of(obj)
        return _name or fallback
    except Exception:  # noqa: BLE001 - issuer name falls back to the query issuer, never raises
        return fallback


def _activity_records_of(filing: object, ticker_or_cik: str | int) -> list[InsiderTransaction]:
    accession = getattr(filing, "accession_no", "")
    form = getattr(filing, "form", "")
    filed_at = getattr(filing, "filed_at", None)
    issuer = getattr(filing, "filer_name", None) or str(ticker_or_cik)
    obj = load_ownership(accession)
    return normalize_ownership_filing(
        obj,
        issuer=_issuer_name_of(obj, issuer),
        form=form,
        filed_at=filed_at,
        accession_no=accession,
        document_name=getattr(filing, "primary_document", None),
        known_at=getattr(filing, "known_at", None) or filed_at,
    )


def get_insider_activity(
    ticker_or_cik: str | int,
    *,
    as_of: str | None = None,
    limit: int | None = 50,
    forms: tuple[str, ...] | list[str] = _DEFAULT_FORMS,
) -> list[InsiderTransaction]:
    filings = list_sec_filings(ticker_or_cik, forms=list(forms), as_of=as_of, limit=limit)
    out: list[InsiderTransaction] = []
    for filing in filings:
        try:
            out.extend(_activity_records_of(filing, ticker_or_cik))
        except Exception:  # noqa: BLE001, S112 - failed filing is skipped, the batch continues
            continue
    if limit is not None:
        out = out[:limit]
    return out


def _columns_of(df: object) -> list[object]:
    try:
        return list(getattr(df, "columns", None) or [])
    except Exception:  # noqa: BLE001 - frame columns degrade to empty on untrusted frame, never raises
        return []


def _share_target_in(columns: list[object]) -> object:
    return next((c for c in columns if "share" in str(c).lower()), None)


def _getitem_of(df: object) -> object:
    try:
        getitem = getattr(df, "__getitem__")  # noqa: B009 - pandas-like frame has no stubs; getattr keeps the dynamic boundary instead of failing the checker
    except Exception:  # noqa: BLE001 - frame getitem read degrades to None on untrusted frame, never raises
        return None
    return getitem if callable(getitem) else None


def _call_getitem(getitem: object, target: object) -> object:
    if not callable(getitem):
        return None
    try:
        return getitem(target)
    except Exception:  # noqa: BLE001 - frame cell read degrades to None on untrusted frame, never raises
        return None


def _column_cells(rows: list[object], target: object) -> list[object]:
    """Non-missing cells for one column target across fallback rows."""
    out: list[object] = []
    for row in rows:
        try:
            cell = (
                row.get(target)
                if isinstance(row, dict)
                else (
                    row[target]
                    if isinstance(row, (list, tuple)) and isinstance(target, int) and 0 <= target < len(row)
                    else None
                )
            )
        except Exception:  # noqa: BLE001, S112 - bad row is skipped, the scan continues
            continue
        if cell is not None:
            out.append(cell)
    return out


def _values_by_column(df: object, target: object) -> object:
    if hasattr(df, "columns"):
        result = _call_getitem(_getitem_of(df), target)
        if result is not None:
            return result
    if not isinstance(df, Iterable) or isinstance(target, bool) or not isinstance(target, (str, int)):
        return None
    try:
        rows: list[object] = list(df)
    except Exception:  # noqa: BLE001 - frame row coercion degrades to None on untrusted frame, never raises
        return None
    return _column_cells(rows, target)


def _values_of_rows(df: object) -> object:
    if not isinstance(df, Iterable):
        return None
    try:
        rows = list(df)
    except Exception:  # noqa: BLE001 - frame iteration degrades to None on untrusted frame, never raises
        return None
    if not rows:
        return None
    first = rows[0]
    if not isinstance(first, dict):
        return None
    target = next((k for k in first if "share" in str(k).lower()), None)
    if target is None:
        return None
    return [row.get(target) for row in rows]


def _total_of(values: object) -> int | None:
    total = 0
    found = False
    try:
        iterator = list(values) if isinstance(values, Iterable) else []
    except Exception:  # noqa: BLE001 - share values degrade to None on untrusted frame, never raises
        return None
    for value in iterator:
        parsed = _safe_int(value)
        if parsed is not None:
            total += parsed
            found = True
    return total if found else None


def _sum_shares_column(df: object) -> int | None:
    """Sum the first 'share'-like column; None when nothing parses."""
    columns = _columns_of(df)
    if columns:
        target = _share_target_in(columns)
        if target is None:
            return None
        return _total_of(_values_by_column(df, target))
    return _total_of(_values_of_rows(df))


def _seller_identity_of(form144: object) -> tuple[str | None, str | None]:
    seller = _str_or_none(_first(form144, "person_selling", "seller_name", "seller", "reporting_owner", "name"))
    seller_cik = _str_or_none(_first(form144, "seller_cik", "person_cik", "cik", "owner_cik"))
    return seller, seller_cik


def _resolved_144_form(form144: object, form: str | None) -> str | None:
    try:
        embedded = str(getattr(form144, "form", "") or "").strip()
        if embedded:
            return form or embedded
    except Exception:  # noqa: BLE001, S110 - missing embedded form keeps the caller form
        pass
    return form


def _proposed_shares_of(form144: object) -> int | None:
    try:
        df = getattr(form144, "securities_to_be_sold", None)
    except Exception:  # noqa: BLE001 - missing securities table degrades to None, never raises
        return None
    return _sum_shares_column(df) if df is not None else None


def _empty_144_sale(*, issuer: str, filed_at: str | None, accession_no: str) -> ProposedInsiderSale:
    return ProposedInsiderSale(
        seller_name=None,
        seller_cik=None,
        issuer=issuer,
        filed_at=filed_at,
        accession_no=accession_no,
        shares_proposed=None,
    )


def _clean_issuer_cik(issuer_cik: str | int | None) -> str | None:
    return str(issuer_cik).strip() if issuer_cik is not None else None


def normalize_144(
    form144: object,
    *,
    issuer: str,
    filed_at: str | None,
    accession_no: str,
    issuer_cik: str | int | None = None,
    form: str | None = None,
    document_name: str | None = None,
    known_at: str | None = None,
) -> ProposedInsiderSale:
    """Build a ProposedInsiderSale; never raises."""
    try:
        seller, seller_cik = _seller_identity_of(form144)
        return ProposedInsiderSale(
            seller_name=seller,
            seller_cik=seller_cik,
            issuer=issuer,
            filed_at=filed_at,
            accession_no=accession_no,
            shares_proposed=_proposed_shares_of(form144),
            issuer_cik=_clean_issuer_cik(issuer_cik),
            form=_resolved_144_form(form144, form),
            document_name=document_name,
            known_at=_known_at_of(known_at, filed_at),
        )
    except Exception:  # noqa: BLE001 - 144 normalization degrades to an empty sale, never raises
        return _empty_144_sale(issuer=issuer, filed_at=filed_at, accession_no=accession_no)


def load_144(accession_no: str) -> object:
    """Live seam: Form144 import stays here; raises on failure."""
    from .documents import get_by_accession_number

    filing = get_by_accession_number(accession_no)
    import edgar

    # edgar ships no Form144 stub; getattr keeps the live seam raising on
    # failure instead of failing the checker on a missing SDK attribute.
    form: object = getattr(edgar, "Form144").from_filing(filing)  # noqa: B009 - edgar ships no Form144 stub; getattr keeps the live seam raising on failure instead of failing the checker
    return form


def get_planned_insider_sales(
    ticker_or_cik: str | int,
    *,
    as_of: str | None = None,
    limit: int | None = 20,
    forms: tuple[str, ...] | list[str] = _DEFAULT_144_FORMS,
) -> list[ProposedInsiderSale]:
    filings = list_sec_filings(ticker_or_cik, forms=list(forms), as_of=as_of, limit=limit)
    out: list[ProposedInsiderSale] = []
    for filing in filings:
        try:
            accession = getattr(filing, "accession_no", "")
            filed_at = getattr(filing, "filed_at", None)
            issuer = getattr(filing, "filer_name", None) or str(ticker_or_cik)
            out.append(
                normalize_144(
                    load_144(accession),
                    issuer=issuer,
                    filed_at=filed_at,
                    accession_no=accession,
                    form=getattr(filing, "form", None),
                    document_name=getattr(filing, "primary_document", None),
                    known_at=getattr(filing, "known_at", None) or filed_at,
                )
            )
        except Exception:  # noqa: BLE001, S112 - failed 144 load is skipped, the batch continues
            continue
    if limit is not None:
        out = out[:limit]
    return out


def _sale_rows_of(transactions: Iterable[InsiderTransaction] | None) -> list[InsiderTransaction]:
    try:
        return list(transactions or [])
    except Exception:  # noqa: BLE001 - sale rows degrade to empty on malformed input, never raises
        return []


def _is_matching_sale(txn: InsiderTransaction, seller: str, filed_at: str | None) -> bool:
    if getattr(txn, "transaction_kind", None) != "open_market_sale":
        return False
    name = getattr(txn, "insider_name", None) or ""
    if str(name).strip().lower() != seller.strip().lower():
        return False
    txn_date = getattr(txn, "transaction_date", None)
    return not (txn_date and filed_at and txn_date < filed_at)


def _executed_sale_shares(rows: list[InsiderTransaction], seller: str, filed_at: str | None) -> int:
    executed = 0
    for txn in rows:
        try:
            if not _is_matching_sale(txn, seller, filed_at):
                continue
            shares = getattr(txn, "shares", None)
            if shares is not None:
                executed += shares
        except Exception:  # noqa: BLE001, S112 - malformed transaction is skipped, the match continues
            continue
    return executed


def _match_note_of(seller: str, proposed_shares: int | None, executed: int) -> str:
    if proposed_shares is not None:
        return (
            f"{seller or 'seller'} proposed {proposed_shares} shares; "
            f"{executed} executed in later open-market Form 4 sales"
        )
    return f"{seller or 'seller'} proposed an unknown quantity; {executed} executed in later open-market Form 4 sales"


def compare_144_to_form4(
    proposed: ProposedInsiderSale, transactions: Iterable[InsiderTransaction] | None
) -> dict[str, object]:
    """Match a 144 proposal against later open-market Form 4 sales."""
    seller = proposed.seller_name or ""
    executed = _executed_sale_shares(_sale_rows_of(transactions), seller, proposed.filed_at)
    return {
        "seller_name": proposed.seller_name,
        "proposed_shares": proposed.shares_proposed,
        "executed_sale_shares": executed,
        "matched": executed > 0,
        "note": _match_note_of(seller, proposed.shares_proposed, executed),
    }


_FORMS_13F = ("13F-HR", "13F-HR/A", "13F-NT", "13F-NT/A")
_FORMS_13F_HOLDINGS = ("13F-HR", "13F-HR/A")
_FORMS_13F_NOTICE = ("13F-NT", "13F-NT/A")


def is_13f_notice(form: object) -> bool:
    """True for 13F-NT notice filings: manager filing with no holdings."""
    try:
        return str(form or "").strip().upper() in _FORMS_13F_NOTICE
    except Exception:  # noqa: BLE001 - form check coerces to False, never raises
        return False


def _13f_security_id(cusip: str | None, isin: str | None) -> str | None:
    c = normalize_cusip(cusip)
    i = normalize_isin(isin)
    if c:
        return f"cusip:{c}"
    if i:
        return f"isin:{i}"
    return None


def _voting_label(sole: int | None = None, shared: int | None = None, non: int | None = None) -> str | None:
    parts: list[str] = []
    for label, value in (("sole", sole), ("shared", shared), ("none", non)):
        try:
            if value is None:
                continue
            parts.append(f"{label}={value}")
        except Exception:  # noqa: BLE001, S112 - unstringable part is skipped, the sid continues
            continue
    return " ".join(parts) or None


def _dict_lookup(row: object, name: str) -> object:
    if isinstance(row, dict):
        try:
            return row.get(name)
        except Exception:  # noqa: BLE001 - untrusted row read coerces to None, never raises
            return None
    return None


def _cell_from_dict(row: object, name: str) -> object:
    return _dict_lookup(row, name)


def _mapping_lookup(row: object, name: str) -> object:
    get = getattr(row, "get", None)
    if not callable(get):
        return None
    try:
        return get(name)
    except Exception:  # noqa: BLE001 - untrusted row read coerces to None, never raises
        return None


def _cell_from_attr(row: object, name: str) -> object:
    try:
        value: object = getattr(row, name, None)
    except Exception:  # noqa: BLE001 - untrusted row read coerces to None, never raises
        return None
    if value is None:
        return _mapping_lookup(row, name)
    narrowed: object = value
    return narrowed


def _holding_cell(row: object, *names: str) -> object:
    for name in names:
        value = _cell_from_dict(row, name) if isinstance(row, dict) else _cell_from_attr(row, name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _holding_identifiers(row: object) -> tuple[str | None, str | None, str | None]:
    cusip = normalize_cusip(_holding_cell(row, "Cusip", "cusip", "CUSIP"))
    isin_norm = normalize_isin(_str_or_none(_holding_cell(row, "Isin", "isin", "ISIN")))
    return cusip, isin_norm, _13f_security_id(cusip, isin_norm)


def _holding_put_call(row: object) -> str | None:
    put_call = _str_or_none(_holding_cell(row, "PutCall", "put_call", "putCall"))
    if put_call is None:
        return None
    return put_call.strip().title() or None


def _holding_shares_prn_type(row: object) -> str | None:
    raw_type = _str_or_none(
        _holding_cell(row, "Type", "SSHPRNAMTTYPE", "SshPrnamtType", "sshPrnamtType", "shares_prn_type")
    )
    if raw_type is None:
        return None
    t = raw_type.strip().upper()
    if t in ("SH", "SHARES"):
        return "SH"
    if t in ("PRN", "PRINCIPAL"):
        return "PRN"
    return None


def _holding_amounts_of(
    cell: Callable[..., object],
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    return (
        _safe_int(cell("SharesPrnAmount", "shares", "sshPrnamt", "share_count")),
        _safe_int(cell("Value", "value", "market_value")),
        _safe_int(cell("SoleVoting", "sole_voting", "Sole")),
        _safe_int(cell("SharedVoting", "shared_voting", "Shared")),
        _safe_int(cell("NonVoting", "non_voting", "None")),
    )


def _holding_names_of(
    cell: Callable[..., object], report_period: str | None
) -> tuple[str | None, str | None, str | None]:
    return (
        _str_or_none(cell("ReportPeriod", "report_period")) or _str_or_none(report_period),
        _str_or_none(cell("Issuer", "issuer_name", "nameOfIssuer", "issuer")),
        _str_or_none(cell("Class", "class_title", "titleOfClass")),
    )


def _holding_managers_of(cell: Callable[..., object], _row: object = None) -> tuple[str | None, str | None]:
    return (
        _str_or_none(
            cell(
                "InvestmentDiscretion",
                "discretion",
                "investment_discretion",
                "investmentDiscretion",
                "INVESTMENTDISCRETION",
            )
        ),
        _str_or_none(cell("OtherManager", "other_manager", "otherManager", "OTHERMANAGER")),
    )


def _holding_identity_of(
    manager_name: str | None, manager_cik: str | int | None, accession_no: str, source_row: int, security_id: str | None
) -> tuple[str | None, str | None, str]:
    from .models import institutional_holding_id

    return (
        _str_or_none(manager_name),
        str(manager_cik).strip() if manager_cik is not None else None,
        institutional_holding_id(accession_no, source_row, security_id),
    )


def _holding_known_at(known_at: str | None, filed_at: str | None) -> str | None:
    try:
        return known_at if isinstance(known_at, str) else str(known_at) if known_at is not None else filed_at
    except Exception:  # noqa: BLE001 - known_at coerces to the filed_at fallback, never raises
        return filed_at


def _holding_row_to_record(
    row: object,
    *,
    manager_name: str | None,
    manager_cik: str | int | None,
    accession_no: str,
    report_period: str | None,
    filed_at: str | None,
    document_name: str | None,
    known_at: str | None,
    source_url: str | None,
    source_row: int,
) -> InstitutionalHolding:
    """One information-table row -> InstitutionalHolding; raises on bad row."""

    def cell(*names: str) -> object:
        return _holding_cell(row, *names)

    cusip, isin_norm, security_id = _holding_identifiers(row)
    shares, value, sole, shared, non = _holding_amounts_of(cell)
    known = _holding_known_at(known_at, filed_at)
    period, issuer_name, class_title = _holding_names_of(cell, report_period)
    discretion, other_manager = _holding_managers_of(cell, row)
    manager, cik, holding_id = _holding_identity_of(manager_name, manager_cik, accession_no, source_row, security_id)
    return InstitutionalHolding(
        manager_name=manager,
        manager_cik=cik,
        accession_no=accession_no,
        source_row=source_row,
        holding_id=holding_id,
        report_period=period,
        issuer_name=issuer_name,
        entity_id=None,
        security_id=security_id,
        class_title=class_title,
        cusip=cusip,
        isin=isin_norm,
        shares=shares,
        value=value,
        put_call=_holding_put_call(row),
        discretion=discretion,
        other_manager=other_manager,
        shares_prn_type=_holding_shares_prn_type(row),
        voting=_voting_label(sole, shared, non),
        filed_at=filed_at,
        known_at=known,
        document_name=document_name,
        source_url=source_url,
    )


def _rows_of_infotable(infotable: object) -> list[object] | None:
    try:
        to_dict = getattr(infotable, "to_dict", None)
        if callable(to_dict):
            raw_rows: object = to_dict(orient="records")
            if isinstance(raw_rows, dict):
                return [raw_rows]
            if isinstance(raw_rows, list):
                return list(raw_rows)
            if isinstance(raw_rows, Iterable):
                return list(raw_rows)
            return []
        if isinstance(infotable, Iterable):
            return list(infotable)
        return []
    except Exception:  # noqa: BLE001 - infotable coercion degrades to None on malformed payload, never raises
        return None


def _clean_manager_cik(manager_cik: str | int | None) -> str | None:
    try:
        return str(manager_cik).strip() if manager_cik is not None else None
    except Exception:  # noqa: BLE001 - manager CIK coerces to None, never raises
        return None


def _append_holding_record(
    out: list[InstitutionalHolding],
    row: object,
    *,
    manager_name: str | None,
    cik: str | None,
    accession_no: str,
    report_period: str | None,
    filed_at: str | None,
    document_name: str | None,
    known_at: str | None,
    source_url: str | None,
    source_row: int,
) -> None:
    try:
        out.append(
            _holding_row_to_record(
                row,
                manager_name=manager_name,
                manager_cik=cik,
                accession_no=accession_no,
                report_period=report_period,
                filed_at=filed_at,
                document_name=document_name,
                known_at=known_at,
                source_url=source_url,
                source_row=source_row,
            )
        )
    except Exception:  # noqa: BLE001, S110 - malformed holding row is skipped, the batch continues
        pass


def normalize_13f_holdings(
    infotable: object,
    *,
    manager_name: str | None = None,
    manager_cik: str | int | None = None,
    accession_no: str,
    report_period: str | None = None,
    filed_at: str | None = None,
    form: str | None = None,
    document_name: str | None = None,
    known_at: str | None = None,
    source_url: str | None = None,
) -> list[InstitutionalHolding]:
    """13F-HR/A information tables -> holdings; NT -> no holdings. Never raises."""

    if is_13f_notice(form):
        return []
    rows = _rows_of_infotable(infotable)
    if rows is None:
        return []
    cik = _clean_manager_cik(manager_cik)
    out: list[InstitutionalHolding] = []
    for source_row, row in enumerate(rows, start=1):
        _append_holding_record(
            out,
            row,
            manager_name=manager_name,
            cik=cik,
            accession_no=accession_no,
            report_period=report_period,
            filed_at=filed_at,
            document_name=document_name,
            known_at=known_at,
            source_url=source_url,
            source_row=source_row,
        )
    return out


def _holding_identifiers_of(holding: InstitutionalHolding) -> tuple[str | None, str | None, str | None]:
    cusip = normalize_cusip(getattr(holding, "cusip", None))
    isin = normalize_isin(getattr(holding, "isin", None))
    return cusip, isin, _13f_security_id(cusip, isin)


def _check_as_of_opt(as_of: str | None) -> str | None:
    """Strict YYYY-MM-DD gate shared by the live query wrappers."""
    if as_of is None:
        return None
    text = str(as_of)
    try:
        date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"invalid as_of date: {as_of!r} (expected YYYY-MM-DD)") from None
    return text


def _known_as_of_record(record: object, as_of: str | None) -> bool:
    """PIT visibility over InsiderTransaction/InstitutionalHolding records."""
    if as_of is None:
        return True
    from .models import pit_of

    value, _basis = pit_of(record)
    return value is not None and value[:10] <= as_of


def _cap(records: list[InsiderTransaction], limit: int) -> list[InsiderTransaction]:
    return records if limit is None else records[:limit]


def _cap_holdings(records: list[InstitutionalHolding], limit: int) -> list[InstitutionalHolding]:
    return records if limit is None else records[:limit]


# Seam: live 13F holdings normalize per filing via SourceGateway + normalization + raw_archive + write_bundle; NOTE: warehouse slots behind live readers.


def _manager_filings(manager_cik: int | str, *, as_of: str | None, limit: int | None) -> list[InstitutionalHolding]:
    """Live 13F-HR/A holdings for one manager CIK via provider filings."""
    from .documents import get_by_accession_number
    from .filings import list_sec_filings as _live_list

    try:
        filings = _live_list(manager_cik, forms=list(_HOLDING_FORMS), as_of=as_of, limit=limit)
    except Exception:
        return []
    out: list[InstitutionalHolding] = []
    for filing in filings:
        try:
            inner = get_by_accession_number(filing.accession_no).obj()  # type: ignore[union-attr]
        except Exception:
            continue
        table = None
        for attr in ("infotable", "information_table", "holdings", "info_table"):
            try:
                table = getattr(inner, attr, None)
            except Exception:
                table = None
            if table is not None:
                break
        if table is None:
            continue
        try:
            out.extend(
                normalize_13f_holdings(
                    table,
                    manager_cik=str(manager_cik),
                    accession_no=filing.accession_no,
                    filed_at=getattr(filing, "filed_at", None),
                    known_at=getattr(filing, "known_at", None),
                )
            )
        except Exception:
            continue
    return [rec for rec in out if _known_as_of_record(rec, as_of)]


def query_issuer_insiders(
    issuer_cik: int | str, *, as_of: str | None = None, root: Path | str | None = None, limit: int = 200
) -> list[InsiderTransaction]:
    """Issuer -> live ownership transactions normalized per filing (PIT in domain)."""
    del root
    bound = _check_as_of_opt(as_of)
    try:
        return _cap(get_insider_activity(issuer_cik, as_of=bound, limit=limit), limit)
    except Exception:
        return []


def query_person_transactions(
    owner_cik: int | str, *, as_of: str | None = None, root: Path | str | None = None, limit: int = 200
) -> list[InsiderTransaction]:
    """Person -> live transactions filtered to one owner CIK (PIT in domain)."""
    del root
    bound = _check_as_of_opt(as_of)
    want = str(owner_cik).strip()
    try:
        txns = get_insider_activity(owner_cik, as_of=bound, limit=limit)
    except Exception:
        return []
    out = [txn for txn in txns if str(getattr(txn, "owner_cik", "") or "").strip() in ("", want)]
    return _cap(out, limit)


def query_manager_holdings(
    manager_cik: int | str, *, as_of: str | None = None, root: Path | str | None = None, limit: int = 200
) -> list[InstitutionalHolding]:
    """Manager -> live 13F holdings normalized per filing (PIT in domain)."""
    del root
    bound = _check_as_of_opt(as_of)
    return _cap_holdings(_manager_filings(manager_cik, as_of=bound, limit=limit), limit)


def _security_matches(holding: InstitutionalHolding, want: str) -> bool:
    cusip, isin, security_id = _holding_identifiers_of(holding)
    candidates = {str(v or "").strip().upper() for v in (cusip, isin, security_id)}
    return want.strip().upper() in candidates


def query_security_managers(
    security: str, *, as_of: str | None = None, root: Path | str | None = None, limit: int = 200
) -> list[InstitutionalHolding]:
    """Security (CUSIP/ISIN/security_id) -> live holdings filtered in domain."""
    del root
    bound = _check_as_of_opt(as_of)
    want = str(security or "").strip().upper()
    if not want:
        return []
    return _cap_holdings(
        [rec for rec in _manager_filings(security, as_of=bound, limit=None) if _security_matches(rec, want)],
        limit,
    )
