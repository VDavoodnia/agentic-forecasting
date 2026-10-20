"""Mini-agents for exp75: one factor agent per macro factor + diagnostics family.

Two agent families:

- **FACTOR agents**: one per Macro Signal Service category (22) plus two
  RAG-sourced supplements (``rent``, ``climate``). Each factor agent is a
  *thin, focused reader of its own category*: it reads the category's
  monthly signal from the ``macro_signals`` table (PIT: signal month must be
  strictly before the origin month — a signal *for* month M synthesises
  news published within M) and emits a narrow :class:`FactorSignal`
  ``{direction, confidence, pressure, reason}``. **No cross-category
  reasoning in the factor agents** — synthesis is the orchestrator's job
  exclusively. Factor signals are month-stamped and cached per month, so
  the marginal cost per origin is only the orchestrator call.
- **DIAGNOSTIC agents**: Stability, Disagreement, Seasonality, Group. They
  speak about the *series/model*, not the world, and are built on the
  deterministic :mod:`td_one.agent.layer2_tools` toolbox.

Direction vocabulary (per the signal service rubric): ``direction`` in
``{-2, -1, 0, 1, 2}`` follows the category's own rubric (e.g. for
``central_bank_rates``: -2 = dovish tilt, +2 = hawkish tilt; for
``labour``: +2 = unemployment rising). ``pressure`` is the *target-relative*
interpretation: +1 means the factor pushes the forecast target UP,
-1 means it pushes it DOWN, scaled by confidence. The pressure-sign maps
(:data:`PRESSURE_ATTRITION`, :data:`PRESSURE_PRICE`) are the analyst's
pre-registered priors (exp75 report) — the family-weight validation and
the ablation decide which ones survive.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

#: The 22 Macro Signal Service categories, exactly as in the macro_signals table.
SIGNAL_CATEGORIES: tuple[str, ...] = (
    "immigration_law",
    "food_prices",
    "basic_needs",
    "income_tax",
    "minimum_wage",
    "salary_region",
    "school_childcare",
    "services_costs",
    "inflation",
    "elections_politics",
    "wars_geopolitics",
    "pandemics_health",
    "flight_prices",
    "car_prices",
    "global_trade",
    "ai_startups",
    "markets_sp500",
    "central_bank_rates",
    "housing",
    "labour",
    "fx",
    "small_business",
)

#: RAG-sourced supplements (not in macro_signals; keyword-heuristic v0 readers).
SUPPLEMENT_CATEGORIES: tuple[str, ...] = ("rent", "climate")

#: Factor families for grouped voting / family-level ablation.
FACTOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "prices": (
        "food_prices",
        "basic_needs",
        "inflation",
        "car_prices",
        "flight_prices",
        "services_costs",
    ),
    "housing": ("housing", "rent", "school_childcare"),
    "labour": ("labour", "minimum_wage", "salary_region", "immigration_law"),
    "policy": ("central_bank_rates", "income_tax", "elections_politics", "small_business"),
    "external": (
        "wars_geopolitics",
        "pandemics_health",
        "global_trade",
        "fx",
        "markets_sp500",
        "ai_startups",
        "climate",
    ),
}

#: Pre-registered pressure-sign priors for ATTRITION-type targets
#: (runoff rate / SMM / switching): +1 = factor pushes attrition UP.
PRESSURE_ATTRITION: dict[str, float] = {
    "central_bank_rates": -1.0,  # hawkish -> lock-in -> runoff down
    "inflation": 1.0,
    "food_prices": 1.0,
    "basic_needs": 1.0,
    "housing": 1.0,  # heat -> equity -> sales/refi up
    "rent": 1.0,
    "income_tax": 1.0,
    "minimum_wage": -1.0,
    "salary_region": -1.0,
    "labour": 1.0,  # + = unemployment rising -> stress up
    "school_childcare": 1.0,
    "services_costs": 1.0,
    "car_prices": 1.0,
    "flight_prices": 1.0,
    "elections_politics": 1.0,
    "wars_geopolitics": 1.0,
    "pandemics_health": 1.0,
    "global_trade": 1.0,
    "fx": -1.0,  # + = CAD strengthening -> cheaper imports
    "markets_sp500": -1.0,  # rally -> wealth -> less stress
    "small_business": 1.0,  # + = deteriorating
    "ai_startups": -1.0,  # boom -> jobs -> less stress
    "immigration_law": 1.0,  # + = tightening -> headwinds
    "climate": 1.0,
}

#: Pre-registered pressure-sign priors for PRICE-type targets (Food CPI):
#: +1 = factor pushes the price index UP.
PRESSURE_PRICE: dict[str, float] = {
    "inflation": 1.0,
    "food_prices": 1.0,
    "fx": -1.0,  # + = CAD strengthening -> cheaper imports
    "labour": -1.0,  # + = unemployment rising -> less wage pressure
    "central_bank_rates": -1.0,  # hawkish -> demand cooling
    "minimum_wage": 1.0,  # cost push
    "salary_region": 1.0,
    "income_tax": -1.0,  # burden -> demand down
    "housing": 1.0,  # shelter spillover
    "basic_needs": 1.0,
    "services_costs": 1.0,
    "car_prices": 1.0,
    "school_childcare": 1.0,
    "flight_prices": 1.0,
    "rent": 1.0,
    "global_trade": 1.0,  # + = barriers -> import prices up
    "wars_geopolitics": 1.0,  # energy pass-through
    "pandemics_health": 1.0,  # supply disruption
    "climate": 1.0,  # crop disruption
    "markets_sp500": 0.0,
    "ai_startups": 0.0,
    "elections_politics": 0.0,
    "immigration_law": 0.0,
    "small_business": 0.0,
}

#: Which pressure map to use per target kind.
PRESSURE_MAPS: dict[str, dict[str, float]] = {
    "attrition": PRESSURE_ATTRITION,
    "price": PRESSURE_PRICE,
}


@dataclass(frozen=True)
class FactorSignal:
    """One factor agent's narrow output."""

    category: str
    family: str
    month: str  # "YYYY-MM" of the signal (strictly < origin month)
    direction: int  # -2..+2 per the category rubric
    confidence: float  # 0..1
    pressure: float  # signed: direction * sign_map * confidence
    trend_3m: float  # direction(now) - direction(3mo ago), /4 -> [-1, 1]
    risk: str  # low | med | high (from the service)
    reason: str  # one sentence
    available: bool = True


@dataclass(frozen=True)
class DiagnosticSignal:
    """One diagnostic agent's narrow output."""

    name: str  # stability | disagreement | seasonality | group
    direction: int  # -1 | 0 | +1 (for stability/disagreement: +1 = MORE uncertainty)
    confidence: float  # 0..1
    pressure: float  # signed pressure on the target (seasonality/group); 0 for pure-U agents
    reason: str  # one sentence
    available: bool = True


def _family_of(category: str) -> str:
    for fam, members in FACTOR_FAMILIES.items():
        if category in members:
            return fam
    return "external"


def _confidence(certainty: float, n_articles: int, freshness_months: int) -> float:
    """Confidence from the service's certainty, coverage, and freshness."""
    cov = min(1.0, n_articles / 5.0)
    fresh = 0.9 ** max(0, freshness_months)
    return float(max(0.0, min(1.0, certainty * cov * fresh)))


class FactorAgent:
    """One thin, focused reader per macro factor category.

    The deterministic primary design: the *reading* was done by the Macro
    Signal Service (LLM, already in ``macro_signals``); this agent
    translates the category's monthly reading into a target-relative
    pressure. Each agent owns exactly one factor — no cross-category
    reasoning.
    """

    def __init__(self, category: str, news_db: str | Path) -> None:
        if category not in SIGNAL_CATEGORIES and category not in SUPPLEMENT_CATEGORIES:
            raise ValueError(f"unknown factor category {category!r}")
        self.category = category
        self.family = _family_of(category)
        self.news_db = Path(news_db)

    # -- primary path: macro_signals table --------------------------------

    def _latest_signal(self, asof_month: str) -> dict[str, Any] | None:
        """Latest signal row with month < asof_month (PIT-safe)."""
        try:
            con = sqlite3.connect(str(self.news_db))
        except sqlite3.Error:
            return None
        try:
            try:
                row = con.execute(
                    "SELECT month, direction, risk, certainty, n_articles "
                    "FROM macro_signals WHERE category = ? AND month < ? "
                    "ORDER BY month DESC LIMIT 1",
                    (self.category, asof_month),
                ).fetchone()
                if row is None:
                    return None
                older = con.execute(
                    "SELECT direction FROM macro_signals WHERE category = ? "
                    "AND month <= date(?, '-3 months') ORDER BY month DESC LIMIT 1",
                    (self.category, row[0]),
                ).fetchone()
            except sqlite3.Error:
                return None  # macro_signals table absent -> degrade
        finally:
            con.close()
        assert row is not None
        return {
            "month": str(row[0]),
            "direction": int(row[1]),
            "risk": str(row[2]),
            "certainty": float(row[3]),
            "n_articles": int(row[4]),
            "direction_3m_ago": int(older[0]) if older else 0,
        }

    def read(self, asof_month: str, target_kind: str = "attrition") -> FactorSignal:
        """Read the factor as of ``asof_month`` ("YYYY-MM").

        Only signals with month < asof_month are visible (PIT). Returns
        an unavailable signal when the category has no history yet.
        """
        if target_kind not in PRESSURE_MAPS:
            raise ValueError(f"target_kind must be one of {sorted(PRESSURE_MAPS)}")
        row = self._latest_signal(asof_month)
        if row is None:
            return FactorSignal(
                category=self.category,
                family=self.family,
                month="",
                direction=0,
                confidence=0.0,
                pressure=0.0,
                trend_3m=0.0,
                risk="low",
                reason=f"{self.category}: no signal history before {asof_month}",
                available=False,
            )
        sign = PRESSURE_MAPS[target_kind].get(self.category, 0.0)
        d = row["direction"]
        fresh = (
            (int(asof_month[:4]) - int(row["month"][:4])) * 12
            + (int(asof_month[5:7]) - int(row["month"][5:7]))
            - 1
        )
        conf = _confidence(row["certainty"], row["n_articles"], max(0, fresh))
        pressure = float((1 if d > 0 else (-1 if d < 0 else 0)) * sign * conf)
        trend = (d - row["direction_3m_ago"]) / 4.0
        reason = (
            f"{self.category}: direction {d:+d}/2 ({row['risk']}, "
            f"conf {conf:.2f}, {row['n_articles']} articles, {row['month']}) "
            f"-> pressure {pressure:+.2f} on {target_kind}"
        )
        return FactorSignal(
            category=self.category,
            family=self.family,
            month=row["month"],
            direction=d,
            confidence=conf,
            pressure=pressure,
            trend_3m=float(trend),
            risk=row["risk"],
            reason=reason,
        )


class FactorCache:
    """Month-stamped JSON cache for factor signals.

    Factor signals are month-stamped, not origin-stamped: all origins in
    the same calendar month share one bundle. The marginal cost per
    origin is therefore only the orchestrator call.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def get(self, asof_month: str, target_kind: str) -> Any:
        return self._data.get(f"{asof_month}|{target_kind}")

    def put(self, asof_month: str, target_kind: str, bundle: Sequence[FactorSignal]) -> None:
        self._data[f"{asof_month}|{target_kind}"] = [
            {
                "category": s.category,
                "family": s.family,
                "month": s.month,
                "direction": s.direction,
                "confidence": s.confidence,
                "pressure": s.pressure,
                "trend_3m": s.trend_3m,
                "risk": s.risk,
                "reason": s.reason,
                "available": s.available,
            }
            for s in bundle
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1))

    def load_bundle(self, asof_month: str, target_kind: str) -> list[FactorSignal] | None:
        raw = self.get(asof_month, target_kind)
        if raw is None:
            return None
        return [FactorSignal(**cast("dict[str, Any]", r)) for r in raw]


def read_all_factors(
    news_db: str | Path,
    asof_month: str,
    target_kind: str = "attrition",
    cache: FactorCache | None = None,
) -> list[FactorSignal]:
    """Read all factor agents as of ``asof_month`` (cache-aware)."""
    if cache is not None:
        hit = cache.load_bundle(asof_month, target_kind)
        if hit is not None:
            return hit
    agents = [FactorAgent(c, news_db) for c in SIGNAL_CATEGORIES]
    bundle = [a.read(asof_month, target_kind) for a in agents]
    # RAG supplements.
    for cat in SUPPLEMENT_CATEGORIES:
        bundle.append(read_supplement_factor(news_db, cat, asof_month, target_kind))
    if cache is not None:
        cache.put(asof_month, target_kind, bundle)
    return bundle


# ---------------------------------------------------------------------------
# RAG supplements (rent, climate): keyword-heuristic v0 readers.
# ---------------------------------------------------------------------------

_SUPPLEMENT_QUERIES: dict[str, str] = {
    "rent": "rent prices rental market Canada",
    "climate": "extreme weather climate floods wildfire Canada",
}

_UP_WORDS = ("rise", "rising", "rose", "record", "surge", "soar", "climb", "increase")
_DOWN_WORDS = ("fall", "falling", "fell", "drop", "decline", "plunge", "decrease", "ease")

#: Module-level cache of semantic indexes (embedding load is expensive).
_INDEX_CACHE: dict[str, Any] = {}


def _supplement_index(news_db: str | Path) -> Any | None:
    key = str(news_db)
    if key not in _INDEX_CACHE:
        try:
            from td_one.rag.semantic import SemanticNewsIndex  # noqa: PLC0415

            _INDEX_CACHE[key] = SemanticNewsIndex(db_path=Path(news_db))
        except Exception:  # noqa: BLE001 — supplements degrade to unavailable
            _INDEX_CACHE[key] = None
    return _INDEX_CACHE[key]


def read_supplement_factor(
    news_db: str | Path,
    category: str,
    asof_month: str,
    target_kind: str = "attrition",
) -> FactorSignal:
    """Heuristic v0 reader for categories absent from macro_signals.

    RAG-retrieves up to 20 recent chunks for the category query (PIT:
    pub_date < asof_month) and scores direction from keyword sentiment.
    Labelled heuristic in the report; ablation decides whether it survives.
    """
    index = _supplement_index(news_db)
    try:
        hits = (
            index.search(_SUPPLEMENT_QUERIES[category], before=asof_month + "-01", top_k=20)
            if index is not None
            else []
        )
    except Exception:  # noqa: BLE001 — supplements must never crash the bundle
        hits = []
    up = down = 0
    for h in hits:
        text = f"{getattr(h, 'title', '')} {getattr(h, 'snippet', '')}".lower()
        up += sum(w in text for w in _UP_WORDS)
        down += sum(w in text for w in _DOWN_WORDS)
    total = up + down
    if total == 0:
        direction = 0
    else:
        score = (up - down) / total
        if score > 0.5:
            direction = 2
        elif score > 0.15:
            direction = 1
        elif score < -0.5:
            direction = -2
        elif score < -0.15:
            direction = -1
        else:
            direction = 0
    sign = PRESSURE_MAPS[target_kind].get(category, 1.0)
    conf = min(1.0, len(hits) / 20.0) * (0.5 if total else 0.0)
    pressure = float((1 if direction > 0 else (-1 if direction < 0 else 0)) * sign * conf)
    return FactorSignal(
        category=category,
        family=_family_of(category),
        month=asof_month,
        direction=direction,
        confidence=conf,
        pressure=pressure,
        trend_3m=0.0,
        risk="low",
        reason=(
            f"{category}: heuristic RAG reader ({len(hits)} chunks, "
            f"kw {up:+d}/-{down}) -> direction {direction:+d}, pressure {pressure:+.2f}"
        ),
        available=total > 0,
    )


# ---------------------------------------------------------------------------
# Diagnostics family (speak about the series/model, not the world).
# ---------------------------------------------------------------------------


def stability_agent(y: np.ndarray, min_n: int = 36) -> DiagnosticSignal:
    """Recent volatility / breaks / error history -> uncertainty up/down.

    +1 (more uncertainty) when the recent window is unstable: fresh
    changepoint, volatility explosion, or unstable level/trend (CUSUM).
    """
    from td_one.agent.layer2_tools import (  # noqa: PLC0415
        changepoint_scan,
        series_diagnostics,
    )

    x = np.asarray(y, dtype=float).ravel()
    if x.size < min_n:
        return DiagnosticSignal("stability", 0, 0.0, 0.0, "stability: series too short", False)
    try:
        diag = series_diagnostics(x)
    except ValueError:
        return DiagnosticSignal("stability", 0, 0.0, 0.0, "stability: diagnostics failed", False)
    cp = changepoint_scan(x)
    votes = 0
    reasons: list[str] = []
    if cp["break_flag"]:
        votes += 1
        reasons.append(f"changepoint (level_z={cp['level_shift_z']:.1f})")
    vol = diag.get("vol_ratio")
    if isinstance(vol, float) and vol == vol and vol > 2.0:
        votes += 1
        reasons.append(f"vol_ratio={vol:.1f}")
    cusum = diag.get("cusum_p")
    if isinstance(cusum, float) and cusum == cusum and cusum < 0.05:
        votes += 1
        reasons.append(f"cusum_p={cusum:.3f}")
    if votes == 0:
        return DiagnosticSignal("stability", 0, 0.6, 0.0, "stability: series stable")
    conf = 0.5 + 0.25 * votes
    return DiagnosticSignal(
        "stability", 1, min(1.0, conf), 0.0, f"stability: unstable — {'; '.join(reasons)}"
    )


def disagreement_agent(candidate_grids: Mapping[str, np.ndarray]) -> DiagnosticSignal:
    """Expert disagreement -> uncertainty up/down.

    Wide disagreement among candidates' step-1 medians (relative to
    their typical scale) means the model choice matters -> more
    uncertainty.
    """
    meds: list[float] = []
    for g in candidate_grids.values():
        arr = np.asarray(g, dtype=float)
        meds.append(float(arr[4, 0]))
    meds_arr = np.asarray(meds)
    scale = float(np.mean(np.abs(meds_arr))) + 1e-9
    cv = float(np.std(meds_arr) / scale)
    if cv > 0.5:
        return DiagnosticSignal(
            "disagreement", 1, min(1.0, 0.5 + cv), 0.0, f"disagreement: median CV={cv:.2f}"
        )
    return DiagnosticSignal(
        "disagreement", 0, 0.6, 0.0, f"disagreement: experts agree (CV={cv:.2f})"
    )


def seasonality_agent(y: np.ndarray, period: int = 12) -> DiagnosticSignal:
    """Seasonal position/strength -> seasonal up/down pressure on the target."""
    from td_one.agent.layer2_tools import series_diagnostics  # noqa: PLC0415

    x = np.asarray(y, dtype=float).ravel()
    if x.size < 36:
        return DiagnosticSignal("seasonality", 0, 0.0, 0.0, "seasonality: series too short", False)
    try:
        diag = series_diagnostics(x, period=period)
    except ValueError:
        return DiagnosticSignal(
            "seasonality", 0, 0.0, 0.0, "seasonality: diagnostics failed", False
        )
    strength = float(diag.get("seasonal_strength", 0.0))
    if strength < 0.3:
        return DiagnosticSignal(
            "seasonality", 0, 0.7, 0.0, f"seasonality: weak (strength={strength:.2f})"
        )
    # Seasonal position: where in the seasonal cycle is the next step?
    resid_centered = x[-period:] - float(np.mean(x[-period:]))
    nxt = float(resid_centered[0])  # seasonal-naive step-1 deviation proxy
    direction = 1 if nxt > 0 else (-1 if nxt < 0 else 0)
    return DiagnosticSignal(
        "seasonality",
        direction,
        min(1.0, strength),
        float(direction * min(1.0, strength)),
        f"seasonality: strength={strength:.2f}, next-step seasonal dev {nxt:+.4f}",
    )


def group_agent(
    group_series: Mapping[str, np.ndarray] | None,
    target: str = "total",
) -> DiagnosticSignal:
    """Cohort/demographic dynamics -> group-level pressure.

    Compares each peer group's recent momentum to the portfolio
    (``target``); if some groups diverge strongly, pressure follows the
    diverging tail. Unavailable when no group series exist.
    """
    if not group_series or target not in group_series:
        return DiagnosticSignal("group", 0, 0.0, 0.0, "group: no group series", False)

    tot = np.asarray(group_series[target], dtype=float).ravel()
    if tot.size < 13:
        return DiagnosticSignal("group", 0, 0.0, 0.0, "group: series too short", False)
    tot_mom = float(tot[-1] - tot[-13])
    devs: list[float] = []
    for name, g in group_series.items():
        if name == target:
            continue
        arr = np.asarray(g, dtype=float).ravel()
        if arr.size < 13:
            continue
        devs.append(float((arr[-1] - arr[-13]) - tot_mom))
    if not devs:
        return DiagnosticSignal("group", 0, 0.0, 0.0, "group: no peer groups", False)
    tail = float(np.max(np.abs(devs)))
    scale = float(np.std(tot[-13:])) + 1e-9
    if tail < 2.0 * scale:
        return DiagnosticSignal("group", 0, 0.6, 0.0, "group: groups track the portfolio")
    direction = 1 if float(np.max(devs)) > 0 else -1
    return DiagnosticSignal(
        "group",
        direction,
        min(1.0, 0.5 + tail / (4.0 * scale)),
        float(direction * min(1.0, 0.5 + tail / (4.0 * scale))),
        f"group: {len(devs)} peers, max divergence {tail:.4f} ({tail / scale:.1f}x scale)",
    )


def read_diagnostics(
    y: np.ndarray,
    candidate_grids: Mapping[str, np.ndarray] | None = None,
    group_series: Mapping[str, np.ndarray] | None = None,
) -> list[DiagnosticSignal]:
    """Run the diagnostics family on visible data."""
    out = [stability_agent(y), seasonality_agent(y)]
    out.append(
        disagreement_agent(candidate_grids)
        if candidate_grids
        else DiagnosticSignal("disagreement", 0, 0.0, 0.0, "disagreement: no candidates", False)
    )
    out.append(group_agent(group_series))
    return out


def family_scores(signals: Sequence[FactorSignal]) -> dict[str, float]:
    """Mean signed pressure per factor family (grouped voting).

    All families are always present (0.0 when no member fires) so the
    vector is comparable across origins.
    """
    acc: dict[str, list[float]] = {fam: [] for fam in FACTOR_FAMILIES}
    for s in signals:
        if s.available and s.family in acc:
            acc[s.family].append(s.pressure)
    return {fam: (float(sum(v) / len(v)) if v else 0.0) for fam, v in acc.items()}
