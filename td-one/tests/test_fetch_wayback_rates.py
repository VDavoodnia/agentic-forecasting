"""Tests for the Wayback/RateHub historical rate parser (mock-only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_wayback_rates import parse_ratehub_html

LEGACY_HTML = """
<table><tbody>
<tr class="rh-analytics"><td><div class="rate"><span>2.59%</span></div></td>
<td>5 year</td><td>Fixed</td>
<td>CanWise Financial A Ratehub Company 4,228 reviews CanWise Financial</td>
</tr>
<tr class="rh-analytics"><td><div class="rate"><span>0.90%</span></div></td>
<td>5 year</td><td>Variable</td>
<td>Meridian Credit Union</td></tr>
</tbody></table>
"""

MODERN_HTML = """
<script id="__NEXT_DATA__" type="application/json">
{"props":{"pageProps":{"bestMarketRates":[
{"term":60,"fixed":{"id":1,"value":3.84,"href":"/best-ontario-mortgage-rate/5-year/fixed/canadian-lender?x=1"},
"variable":{"id":2,"value":3.95,"href":"/best-ontario-mortgage-rate/5-year/variable/eq-bank?x=1"}},
{"term":36,"fixed":{"id":3,"value":3.99,"href":"/r/3-year/fixed/mcap?x=1"},"variable":null},
{"term":12,"fixed":null,"variable":null}
],"bestBankRates":[]}}}
</script>
"""


def test_parse_legacy_ratehub_rows() -> None:
    rows = parse_ratehub_html(LEGACY_HTML)
    assert len(rows) == 2
    fix = [r for r in rows if r["kind"] == "fixed"][0]
    assert fix["term_years"] == 5
    assert fix["rate_pct"] == 2.59
    assert "CanWise" in fix["lender"]
    var = [r for r in rows if r["kind"] == "variable"][0]
    assert var["rate_pct"] == 0.90
    assert var["lender"] == "Meridian Credit Union"


def test_parse_legacy_lender_name_not_duplicated() -> None:
    rows = parse_ratehub_html(LEGACY_HTML)
    fix = [r for r in rows if r["kind"] == "fixed"][0]
    # The "4,228 reviews CanWise Financial" duplicate must not appear twice.
    assert fix["lender"].count("CanWise") == 1
    assert "reviews" not in fix["lender"]


def test_parse_modern_next_data() -> None:
    rows = parse_ratehub_html(MODERN_HTML)
    by_term = {(r["term_years"], r["kind"]): r for r in rows}
    assert by_term[(5, "fixed")]["rate_pct"] == 3.84
    assert by_term[(5, "variable")]["rate_pct"] == 3.95
    assert by_term[(3, "fixed")]["rate_pct"] == 3.99
    assert by_term[(3, "fixed")]["lender"] == "mcap"
    assert (1, "fixed") not in by_term  # null value rows skipped


def test_parse_unknown_layout_returns_empty() -> None:
    assert parse_ratehub_html("<html><body>nothing here</body></html>") == []


def test_sanity_range_rejects_absurd() -> None:
    html = LEGACY_HTML.replace("2.59%", "99.9%")
    rows = parse_ratehub_html(html)
    assert all(r["rate_pct"] <= 15.0 for r in rows)
