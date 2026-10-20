"""exp74 Phase 2b: story-pipeline signals for Food CPI origins (UQ features).

Runs exp69c's StoryPipeline (retrieve -> rank -> story -> classify) on
monthly Food CPI origins 2021-01..2025-12 and stores the parsed ShiftSignal
dicts in ``experiments/exp74_story_signals.json`` — the committed evidence.
The raw LLM prompt/response cache stays local (``exp74_story_llm_cache.json``,
uncommitted, per repo convention).

Resume-safe: origins already present in the output JSON are skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OUT = HERE / "exp74_story_signals.json"
LLM_CACHE = HERE / "exp74_story_llm_cache.json"
DB_PATH = ROOT / "data" / "news" / "news_2026.db"

QUERIES = [
    "Bank of Canada interest rate hike inflation outlook",
    "food prices grocery inflation Canada rising",
    "central bank monetary policy tightening",
]
QUESTION = "Will food-price inflation in Canada shift materially in the next 6 months?"
TARGET_DESC = "Canada Food CPI year-over-year inflation rate (fraction, e.g. 0.05 = 5%)"
HORIZON = 6


def month_range() -> list[str]:
    out = []
    for year in range(2021, 2026):
        for m in range(1, 13):
            out.append(f"{year:04d}-{m:02d}")
    return out


class CachingClient:
    """Disk-cached LLM client (resume-safe; sha256 prompt keys)."""

    def __init__(self, inner, cache_path: Path) -> None:
        import hashlib  # noqa: PLC0415

        self._inner = inner
        self._path = cache_path
        self._hash = hashlib.sha256
        self.llm_calls = 0
        self._mem: dict[str, str] = {}
        if cache_path.exists():
            self._mem = json.loads(cache_path.read_text())

    def complete(self, prompt: str, **kwargs) -> str:
        key = self._hash(prompt.encode()).hexdigest()
        if key not in self._mem:
            self._mem[key] = str(self._inner.complete(prompt, **kwargs))
            self.llm_calls += 1
            if self.llm_calls % 5 == 0:
                self._path.write_text(json.dumps(self._mem))
        return self._mem[key]


def main() -> None:
    from td_one.agent.story_pipeline import StoryPipeline  # noqa: PLC0415
    from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: PLC0415
    from td_one.rag.semantic import SemanticNewsIndex  # noqa: PLC0415

    signals: dict[str, dict] = {}
    if OUT.exists():
        signals = json.loads(OUT.read_text())

    client = CachingClient(VectorProxyClient(model="flash-lite", call_budget=500), LLM_CACHE)
    index = SemanticNewsIndex(db_path=DB_PATH)
    pipe = StoryPipeline(
        client=client,
        index=index,
        queries=QUERIES,
        question=QUESTION,
        target_desc=TARGET_DESC,
        horizon=HORIZON,
    )

    for ym in month_range():
        if ym in signals:
            continue
        origin_ymd = f"{ym}-15"
        before_ymd = f"{ym}-28"
        signal, dbg = pipe.run(
            before_ymd=before_ymd,
            origin_ymd=origin_ymd,
            window_start_ym=ym,
            window_end_ym=ym,
            use_timeline=True,
        )
        signals[ym] = dict(dbg.raw_signal)
        signals[ym]["n_retrieved"] = dbg.n_retrieved
        OUT.write_text(json.dumps(signals))
        print(
            f"{ym}: {signals[ym]['shift_type']}/{signals[ym]['direction']}/"
            f"{signals[ym]['magnitude_bucket']} (llm_calls={client.llm_calls})",
            flush=True,
        )

    print(f"done: {len(signals)} origins, {client.llm_calls} new LLM calls", flush=True)


if __name__ == "__main__":
    main()
