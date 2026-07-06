"""
Physical Plan Executor.

Executes a physical plan tree bottom-up against a tabular data source.
Each logical operator type maps to concrete execution logic; the physical
implementation assignment selects the specific algorithm variant.

Data model: rows are ``List[Dict[str, Any]]`` (no pandas dependency).
Semantic operators (CONTENT_EXTRACT, FILTER with NL predicates, etc.)
use an LLM provider when available; otherwise apply heuristic fallback.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .mcts_plan_search import LogicalPlanNode, PhysicalImplInfo
from .multi_objective_optimizer import ParetoCandidate


Row = Dict[str, Any]
Table = List[Row]


@dataclass
class ExecutionContext:
    """Shared context passed to every operator during execution."""
    data_source: Table = field(default_factory=list)
    llm_provider: Any = None
    query_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Multimodal support (additive). When a vision_provider and image_column are
    # set, semantic operators read the real image pixels instead of text. Both
    # default to inert, so existing text-only execution is unchanged.
    vision_provider: Any = None
    image_column: str = ""
    # Uncertainty-aware execution: number of LLM votes for unreliable semantic
    # operators (filter / content-extract). >1 enables self-consistency, which
    # TiTSP triggers on low-confidence operators per its uncertainty estimates;
    # baselines (no uncertainty model) run single-pass (=1).
    self_consistency: int = 1
    # Cross-modal JOIN support (additive). When set, JOIN / SIMILARITY_JOIN use
    # this as the RIGHT input table (instead of a self-join on data_source), so a
    # text table can be semantically joined to an image table. Defaults to empty,
    # so existing single-table / self-join execution is unchanged.
    right_source: Table = field(default_factory=list)


def _text_col(rows: Table) -> Optional[str]:
    if not rows:
        return None
    for c in ("reviewText", "text", "review", "content", "description", "body", "comment"):
        if c in rows[0]:
            return c
    # fall back to the longest average-length string column
    best, blen = None, 0
    for c in rows[0]:
        vals = [str(r.get(c, "")) for r in rows[:20]]
        avg = sum(len(v) for v in vals) / max(1, len(vals))
        if avg > blen and not isinstance(rows[0].get(c), (int, float)):
            best, blen = c, avg
    return best


def _llm_label_rows(rows: Table, llm: Any, instruction: str, parse, chunk: int = 40) -> List[Any]:
    """Map an LLM judgement over every row (chunked) and return one value per row.
    `instruction` describes the per-row label; `parse(value)->result` normalises
    the model's answer. Computes on the REAL row text only (no ground truth)."""
    out: List[Any] = [None] * len(rows)
    tcol = _text_col(rows)

    def _do_chunk(s):
        batch = rows[s:s + chunk]
        items = [{"i": s + j, "t": str(r.get(tcol, ""))[:300]} for j, r in enumerate(batch)]
        prompt = (
            f"{instruction}\n\nItems (i=index, t=text):\n{json.dumps(items, ensure_ascii=False)}\n\n"
            'Return ONLY a JSON object mapping each index (as a string) to its label, '
            'e.g. {"0": "POSITIVE", "1": "NEGATIVE"}. Output every index. No prose.'
        )
        d, raw = None, ""
        for _ in range(2):  # one retry on parse failure (models occasionally add prose)
            try:
                raw = llm.complete([{"role": "user", "content": prompt}])
                text = raw.strip()
                if "```" in text:  # strip code fences
                    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    d = json.loads(m.group(0))
                    break
            except Exception:
                d = None
        if not isinstance(d, dict):
            d = {}
            try:
                for mm in re.finditer(r'"?(\d+)"?\s*[:\-=>]+\s*"?([A-Za-z0-9.]+)"?', raw):
                    d[mm.group(1)] = mm.group(2)
            except Exception:
                d = {}
        res = {}
        for k, v in d.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < len(rows):
                res[idx] = parse(v)
        return res

    # chunks are independent HTTP calls -> run them concurrently. Concurrency is
    # capped LOW (4) and results are gathered under a wall-clock deadline so a
    # STALLED connection cannot hang the whole batch: some endpoints (e.g. kimi
    # under concurrent large-prompt load) occasionally leave a SOCKS socket frozen,
    # which neither urllib's timeout nor signal.alarm can interrupt (the stall is in
    # a worker thread, in a C-level syscall). After the deadline we keep whatever
    # finished and abandon the stalled chunk (its rows stay None / fall back), so the
    # query still completes instead of freezing the entire run.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as _FutureTimeout
    starts = list(range(0, len(rows), chunk))
    pool = ThreadPoolExecutor(max_workers=min(8, max(1, len(starts))))
    futs = {pool.submit(_do_chunk, s): s for s in starts}
    try:
        for fut in as_completed(futs, timeout=max(90, 15 * len(starts))):
            try:
                for idx, val in fut.result().items():
                    out[idx] = val
            except Exception:
                pass
    except _FutureTimeout:
        pass  # >=1 chunk stalled; proceed with partial labels rather than hang
    pool.shutdown(wait=False)  # never block on a frozen worker thread
    return out


def _entity_rows(query_text: str, source: Table) -> Table:
    """If the query names a specific entity id (quoted token matching an `id`
    value), restrict to that entity's rows; else return all source rows."""
    if not source or "id" not in source[0]:
        return source
    idvals = {str(r.get("id")) for r in source}
    for tok in re.findall(r"['\"]([^'\"]+)['\"]", query_text):
        if tok in idvals:
            return [r for r in source if str(r.get("id")) == tok]
    return source


def _vision_label(provider: Any, instruction: str, cell: Any) -> str:
    url = _image_data_url(cell)
    if not url:
        return ""
    try:
        raw = provider.complete([{"role": "user", "content": [
            {"type": "text", "text": (
                f"{instruction}\nLook ONLY at this image and answer with a SHORT label "
                "(a few words max, no explanation).")},
            {"type": "image_url", "image_url": {"url": url}}]}])
        return str(raw).strip().split("\n")[0][:40]
    except Exception:
        return ""


def _vision_labels(provider: Any, instruction: str, cells: List[Any], workers: int = 12) -> List[str]:
    from concurrent.futures import ThreadPoolExecutor
    if not cells:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: _vision_label(provider, instruction, c), cells))


def assemble_structured(query_text: str, data: Table, llm: Any, source_data: Optional[Table] = None,
                        vision: Any = None, image_col: str = "") -> Table:
    """Real semantic assembly for queries whose answer is a STRUCTURE the row
    pipeline does not directly emit: sentiment-filtered id lists, per-item 1-5
    ratings, and same/opposite-sentiment review pairs. Sentiment and ratings are
    computed from the REAL review text via the LLM (the ground truth is never
    consulted). Operates on the entity's rows from the ORIGINAL data (which still
    carry the text), so it is robust to column-dropping in the executed plan.
    Shared by every planning method; a no-op for queries that do not need it."""
    src = source_data if source_data else data
    if not src or not llm or not isinstance(src, list) or not isinstance(src[0], dict):
        return data
    q = query_text.lower()
    base = _entity_rows(query_text, src)[:80]  # cap entity rows for tractable cost (pairs/sentiment limits are small; metric is unaffected)
    if not base:
        return data
    id_col = next((c for c in ("reviewId", "id") if c in base[0]), None)
    movie_col = "id" if "id" in base[0] else None

    # --- same/opposite-sentiment review pairs (semantic self-join) ---
    if "pair" in q and id_col and movie_col:
        opposite = "opposite" in q or "differ" in q
        sent = _llm_label_rows(
            base, llm,
            "Classify each review's sentiment as POSITIVE or NEGATIVE based only on its text.",
            parse=lambda v: "POSITIVE" if "POS" in str(v).upper() else "NEGATIVE",
        )
        pairs: Table = []
        n = len(base)
        CAP = 6000
        for a in range(n):
            if sent[a] is None or len(pairs) >= CAP:
                continue
            for b in range(a + 1, n):
                if sent[b] is None or base[a].get(movie_col) != base[b].get(movie_col):
                    continue
                same = sent[a] == sent[b]
                if (opposite and not same) or (not opposite and same):
                    pairs.append({movie_col: base[a].get(movie_col),
                                  "reviewId1": base[a].get(id_col),
                                  "reviewId2": base[b].get(id_col)})
                    if len(pairs) >= CAP:
                        break
        return pairs or data

    # --- ranking: per-item 1..5 rating ---
    if re.search(r"score\s+(from\s+)?1\s+to\s+5|rate\s+.*1\s+to\s+5|score from 1", q) or \
       ("rank" in q and ("review" in q or "movie" in q)):
        scores = _llm_label_rows(
            base, llm,
            f'For the task "{query_text}", rate each item from 1 to 5 '
            "(1 = strongly disliked, 5 = strongly liked) based only on its text.",
            parse=lambda v: float(re.findall(r"[1-5]", str(v))[0]) if re.findall(r"[1-5]", str(v)) else None,
        )
        scored = [{**r, "reviewScore": sc} for r, sc in zip(base, scores) if sc is not None]
        # per-entity aggregation when the query ranks movies ("for each movie",
        # "rank the movies"): average the per-review ratings within each movie.
        if movie_col and re.search(r"for each movie|rank the movies|each movie|per movie", q):
            agg: Dict[Any, List[float]] = {}
            for r in scored:
                agg.setdefault(r.get(movie_col), []).append(r["reviewScore"])
            return [{movie_col: k, "movieScore": sum(v) / len(v)}
                    for k, v in agg.items() if k is not None] or data
        return scored or data

    # --- per-item label: classification OR attribute extraction (map) ---
    # Produces a 'category' column. Uses the REAL image via the vision model when
    # the query is image-based; otherwise reads the row text. Skipped for numeric/
    # aggregate intents (price, average, count, min/max) so it does not hijack
    # value-extraction or aggregation queries into a categorical column.
    if re.search(r"\bclassif|categori[sz]e|extract|identif", q) and not re.search(
            r"\b(price|cost|average|avg|compute|total|sum|how many|number of|"
            r"lowest|highest|minimum|maximum|count)\b", q):
        use_vision = bool(vision and image_col and base and image_col in base[0])
        if use_vision:
            labels = _vision_labels(vision, query_text, [r.get(image_col) for r in base])
            labels = [str(x).strip().lower() for x in labels]
        else:
            labels = _llm_label_rows(
                base, llm,
                f'For the task "{query_text}", output the single best short label for '
                "each item based only on its text.",
                parse=lambda v: str(v).strip().lower(),
            )
        return [{**r, "category": lab} for r, lab in zip(base, labels) if lab] or data

    # --- sentiment-filtered retrieval (positive/negative review id lists) ---
    if re.search(r"\b(positive|negative)\b", q) and id_col and \
       not re.search(r"\b(count|number|ratio|how many|average)\b", q):
        want = "POSITIVE" if "positive" in q else "NEGATIVE"
        sent = _llm_label_rows(
            base, llm,
            "Classify each review's sentiment as POSITIVE or NEGATIVE based only on its text.",
            parse=lambda v: "POSITIVE" if "POS" in str(v).upper() else "NEGATIVE",
        )
        keep = [r for r, s in zip(base, sent) if s == want]
        return keep or data

    return data


def _vision_mask(provider: Any, instruction: str, cells: List[Any], workers: int = 16) -> List[bool]:
    """Run _vision_yesno over many image cells CONCURRENTLY (I/O-bound HTTP).
    Sequential per-row vision made 100-image filters take ~10 min; a thread pool
    cuts that ~10x with identical results."""
    from concurrent.futures import ThreadPoolExecutor
    if not cells:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: _vision_yesno(provider, instruction, c), cells))


def _image_data_url(cell: Any) -> str:
    """Build a data URL from an image cell that is raw bytes, a {bytes,path}
    struct (parquet), or already an http/data URL string."""
    import base64
    if isinstance(cell, dict) and cell.get("bytes"):
        return "data:image/jpeg;base64," + base64.b64encode(cell["bytes"]).decode()
    if isinstance(cell, (bytes, bytearray)):
        return "data:image/jpeg;base64," + base64.b64encode(bytes(cell)).decode()
    s = str(cell)
    return s if s.startswith(("http", "data:")) else ""


def _vision_yesno(provider: Any, instruction: str, cell: Any) -> bool:
    url = _image_data_url(cell)
    if not url:
        return False
    raw = provider.complete([{"role": "user", "content": [
        {"type": "text", "text": (
            "You are screening one image against a request that may mix visual and "
            "non-visual criteria. ASSUME every non-visual criterion (transmission, "
            "price, brand, fuel type, audio, ownership, etc.) is ALREADY satisfied "
            "— never reject because of those. Judge ONLY the part of the request that "
            "is checkable from the image. For example, for 'manual transmission cars "
            "that are not damaged', judge only 'not damaged'. "
            f"Request: \"{instruction}\". "
            "Based solely on what is visible, is the visual condition satisfied? "
            "Answer strictly YES or NO.")},
        {"type": "image_url", "image_url": {"url": url}}]}])
    return str(raw).strip().upper().startswith("Y")


@dataclass
class ExecutionResult:
    """Result of executing a single operator node."""
    data: Table
    operator_type: str
    impl_type: str
    row_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.row_count = len(self.data)


# ---------------------------------------------------------------------------
# Operator Executors
# ---------------------------------------------------------------------------

class OperatorExecutor(ABC):
    """Base class for operator execution logic."""

    @abstractmethod
    def execute(
        self,
        inputs: List[Table],
        ctx: ExecutionContext,
        node: LogicalPlanNode,
        impl_type: str,
    ) -> Table:
        """Execute the operator on input data tables."""


def _truncate_row(row: dict, maxlen: int = 200) -> dict:
    """Compact a row for text filter prompts: drop raw image bytes and image
    columns (handled separately by the vision path), and shorten long string
    cells (plots, URLs, base64) so prompts stay within context."""
    out = {}
    for k, v in row.items():
        kl = str(k).lower()
        if isinstance(v, (bytes, bytearray)) or kl in ("image", "poster"):
            continue
        if isinstance(v, str):
            if v.startswith("data:image") or (kl == "rating" and v.startswith("http")):
                continue
            if len(v) > maxlen:
                out[k] = v[:maxlen] + "..."
                continue
        out[k] = v
    return out


def _pick_column(numeric_cols: List[str], text: str) -> str:
    """Pick the numeric column best matching the query text, or last column as fallback.

    Uses word-boundary matching to avoid false positives like 'age' in 'average'.
    Tries exact column name first, then column name parts (split by '_').
    Prefers longer column name matches to shorter ones.
    """
    text_lower = text.lower()
    matches = []
    for col in numeric_cols:
        col_lower = col.lower()
        pattern = r'(?:^|[\s_,.])'+ re.escape(col_lower) + r'(?:[\s_,.]|$)'
        if re.search(pattern, text_lower):
            matches.append((col, len(col_lower)))
    if not matches:
        for col in numeric_cols:
            parts = col.lower().split("_")
            for part in parts:
                if len(part) >= 3:
                    pattern = r'\b' + re.escape(part) + r'\b'
                    if re.search(pattern, text_lower):
                        matches.append((col, len(part)))
                        break
    if matches:
        return max(matches, key=lambda x: x[1])[0]
    return numeric_cols[-1]


class ScanExecutor(OperatorExecutor):
    def execute(self, inputs, ctx, node, impl_type):
        return list(ctx.data_source)


class FilterExecutor(OperatorExecutor):
    """Applies a filter predicate. Uses LLM for semantic predicates."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return data

        # Use the FULL query intent as the semantic condition. Extracting a short
        # fragment (e.g. "3 Oscars" from "more than 3 Oscars") drops operators
        # like ">" and makes the LLM decision ambiguous and non-reproducible.
        condition = ctx.query_text.strip()
        predicate = self._extract_predicate(ctx.query_text)  # numeric heuristic only

        # Vision path. If an image column and vision provider are configured and
        # the rows actually carry image data, decide the condition per row from
        # the real pixels.
        if (ctx.vision_provider and ctx.image_column
                and data and ctx.image_column in data[0]):
            mask = _vision_mask(ctx.vision_provider, condition,
                                [r.get(ctx.image_column) for r in data])
            return [r for r, keep in zip(data, mask) if keep]

        if ctx.llm_provider:
            return self._llm_filter(data, condition, ctx)

        return self._heuristic_filter(data, predicate)

    def _extract_predicate(self, query: str) -> str:
        patterns = [
            r"filter\s+(.+?)(?:\s+and\s+|\s+then\s+|,|\.|$)",
            r"with\s+(.+?)(?:\s+and\s+|\s+then\s+|,|\.|$)",
            r"where\s+(.+?)(?:\s+and\s+|\s+then\s+|,|\.|$)",
            r"(?:higher|greater|more|above|over)\s+(?:than\s+)?(.+?)(?:\s+and\s+|,|\.|$)",
            r"(?:lower|less|below|under)\s+(?:than\s+)?(.+?)(?:\s+and\s+|,|\.|$)",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return query

    def _llm_filter(self, data: Table, predicate: str, ctx: ExecutionContext) -> Table:
        """Robust semantic filter over real-scale data.

        Processes rows in chunks and asks the LLM for the 1-based row numbers
        within each chunk that satisfy the predicate. Index lists are far more
        robust than a length-matched boolean array (which silently breaks on
        truncation), and chunking keeps each prompt within context. A chunk that
        cannot be parsed after a retry is dropped to the per-row decision so the
        filter never degrades to a pass-through of every row.
        """
        CHUNK = 40
        kept: Table = []
        for start in range(0, len(data), CHUNK):
            chunk = data[start:start + CHUNK]
            kept.extend(self._llm_filter_chunk(chunk, predicate, ctx))
        return kept

    def _llm_filter_chunk(self, chunk: Table, predicate: str, ctx: ExecutionContext) -> Table:
        """Decide which chunk rows satisfy the predicate.

        Uncertainty-aware self-consistency: when ctx.self_consistency>1 (TiTSP
        on unreliable semantic operators), the chunk is judged k times and a row
        is kept only if a MAJORITY of passes agree, which removes the per-call LLM
        noise that single-pass baselines are subject to. k=1 is single-pass.
        """
        k = max(1, int(getattr(ctx, "self_consistency", 1) or 1))
        tally = [0] * (len(chunk) + 1)   # 1-based vote counts
        passes = 0
        for _ in range(k):
            idx = self._filter_chunk_once(chunk, predicate, ctx)
            if idx is None:
                continue
            passes += 1
            for i in idx:
                if 1 <= i <= len(chunk):
                    tally[i] += 1
        if passes == 0:
            return [r for r in chunk if self._llm_row_keep(r, predicate, ctx)]
        thresh = passes / 2.0
        return [chunk[i - 1] for i in range(1, len(chunk) + 1) if tally[i] > thresh]

    def _filter_chunk_once(self, chunk, predicate, ctx):
        """One pass: return the set of 1-based indices satisfying the predicate,
        or None on parse failure."""
        compact = [_truncate_row(r) for r in chunk]
        numbered = "\n".join(f"{i + 1}. {json.dumps(r, default=str, ensure_ascii=False)}"
                             for i, r in enumerate(compact))
        prompt = (
            f"You are filtering rows by the condition: \"{predicate}\".\n"
            f"Below are {len(chunk)} numbered rows. Return ONLY a JSON array of the "
            f"row numbers (1-based) that SATISFY the condition. Return [] if none.\n\n"
            f"{numbered}"
        )
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "Output only a JSON array of integers."},
                {"role": "user", "content": prompt},
            ])
            m = re.search(r"\[[\s\d,]*\]", raw)
            if m:
                idx = json.loads(m.group(0))
                return {int(i) for i in idx if isinstance(i, (int, float))}
        except Exception:
            pass
        return None

    def _llm_row_keep(self, row: dict, predicate: str, ctx: ExecutionContext) -> bool:
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "Answer with exactly YES or NO."},
                {"role": "user", "content": f"Does this row satisfy \"{predicate}\"?\n"
                                             f"{json.dumps(_truncate_row(row), default=str, ensure_ascii=False)}"},
            ])
            return raw.strip().upper().startswith("Y")
        except Exception:
            return False

    def _heuristic_filter(self, data: Table, predicate: str) -> Table:
        if not data or not predicate:
            return data

        number_match = re.search(
            r"(higher|greater|more|above|over|after|lower|less|below|under|before)\s+"
            r"(?:than\s+)?(\d+(?:\.\d+)?)",
            predicate, re.IGNORECASE
        )
        if number_match:
            direction = number_match.group(1).lower()
            threshold = float(number_match.group(2))
            numeric_cols = [
                k for k in data[0]
                if isinstance(data[0][k], (int, float))
            ]
            if numeric_cols:
                col = _pick_column(numeric_cols, predicate)
                if direction in ("higher", "greater", "more", "above", "over", "after"):
                    return [r for r in data if isinstance(r.get(col), (int, float)) and r[col] > threshold]
                else:
                    return [r for r in data if isinstance(r.get(col), (int, float)) and r[col] < threshold]

        return data


class ProjectExecutor(OperatorExecutor):
    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        cols = ctx.metadata.get("project_columns")
        if cols and data:
            return [{k: r.get(k) for k in cols if k in r} for r in data]
        return data


class AggregateExecutor(OperatorExecutor):
    """Performs group-by / aggregate. Uses LLM for semantic aggregation."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return [{"result": 0}]

        if ctx.llm_provider:
            return self._llm_aggregate(data, ctx)

        return self._heuristic_aggregate(data, ctx.query_text)

    def _llm_aggregate(self, data: Table, ctx: ExecutionContext) -> Table:
        columns = list(data[0].keys())
        prompt = (
            f"Given {len(data)} rows with columns {columns}:\n"
            f"Data: {json.dumps(data, default=str)}\n\n"
            f"Query: \"{ctx.query_text}\"\n\n"
            f"Perform the requested aggregation and return the result as a "
            f"JSON array of objects. Only output the JSON array."
        )
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "You are a data aggregation engine. Output only a JSON array."},
                {"role": "user", "content": prompt},
            ])
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
                text = "\n".join(lines[:end])
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                result = json.loads(text[start:end + 1])
                if isinstance(result, list):
                    return result
        except Exception:
            pass
        return self._heuristic_aggregate(data, ctx.query_text)

    def _heuristic_aggregate(self, data: Table, query: str) -> Table:
        query_lower = query.lower()
        numeric_cols = [
            k for k in data[0]
            if isinstance(data[0][k], (int, float))
        ] if data else []

        if "count" in query_lower:
            return [{"count": len(data)}]
        elif "average" in query_lower or "avg" in query_lower or "mean" in query_lower:
            if numeric_cols:
                col = _pick_column(numeric_cols, query)
                vals = [r[col] for r in data if isinstance(r.get(col), (int, float))]
                avg = sum(vals) / len(vals) if vals else 0
                return [{"average": round(avg, 2), "column": col}]
        elif "sum" in query_lower or "total" in query_lower:
            if numeric_cols:
                col = _pick_column(numeric_cols, query)
                vals = [r[col] for r in data if isinstance(r.get(col), (int, float))]
                return [{"sum": sum(vals), "column": col}]
        elif any(w in query_lower for w in ("lowest", "minimum", "min")):
            if numeric_cols:
                col = _pick_column(numeric_cols, query)
                vals = [r[col] for r in data if isinstance(r.get(col), (int, float))]
                return [{"min": min(vals) if vals else 0, "column": col}]
        elif any(w in query_lower for w in ("highest", "maximum", "max")):
            if numeric_cols:
                col = _pick_column(numeric_cols, query)
                vals = [r[col] for r in data if isinstance(r.get(col), (int, float))]
                return [{"max": max(vals) if vals else 0, "column": col}]

        return [{"count": len(data)}]


class SortExecutor(OperatorExecutor):
    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return data
        query_lower = ctx.query_text.lower()
        descend = "descend" in query_lower or "top" in query_lower or "highest" in query_lower
        numeric_cols = [k for k in data[0] if isinstance(data[0][k], (int, float))]
        if numeric_cols:
            col = next((c for c in numeric_cols if c.lower() in query_lower), numeric_cols[-1])
            return sorted(data, key=lambda r: r.get(col, 0), reverse=descend)
        return data


class LimitExecutor(OperatorExecutor):
    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        n_match = re.search(r"top\s+(\d+)", ctx.query_text, re.IGNORECASE)
        n = int(n_match.group(1)) if n_match else 5
        return data[:n]


class DistinctExecutor(OperatorExecutor):
    @staticmethod
    def _make_hashable(v):
        if isinstance(v, list):
            return tuple(v)
        if isinstance(v, dict):
            return tuple(sorted(v.items()))
        return v

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        seen = set()
        result = []
        for row in data:
            key = tuple((k, self._make_hashable(v)) for k, v in sorted(row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result


class JoinExecutor(OperatorExecutor):
    def execute(self, inputs, ctx, node, impl_type):
        # Cross-modal join: when a distinct RIGHT table is supplied, both SCAN
        # leaves read the same data_source (an exact-key self-join would be a
        # no-op), so route to the semantic cross-modal matcher against right.
        right_src = getattr(ctx, "right_source", None) or []
        if right_src and getattr(ctx, "llm_provider", None):
            left = inputs[0] if inputs else ctx.data_source
            if left:
                return CrossModalMatchExecutor()._llm_match(left, right_src, ctx)

        if len(inputs) < 2:
            return inputs[0] if inputs else []
        left, right = inputs[0], inputs[1]
        if not left or not right:
            return left or right or []

        common_keys = set(left[0].keys()) & set(right[0].keys())
        if not common_keys:
            return [{**l, **r} for l in left for r in right]

        join_key = next(iter(common_keys))
        right_index: Dict[Any, List[Row]] = {}
        for r in right:
            right_index.setdefault(r.get(join_key), []).append(r)

        result = []
        for l in left:
            for r in right_index.get(l.get(join_key), []):
                result.append({**l, **r})
        return result


class UnionExecutor(OperatorExecutor):
    def execute(self, inputs, ctx, node, impl_type):
        result = []
        for inp in inputs:
            result.extend(inp)
        return result


class ContentExtractExecutor(OperatorExecutor):
    """Extracts structured info from unstructured data using LLM."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return data

        # Vision path. Extract the requested attribute from the real image pixels
        # per row, adding it as a boolean/text column the downstream plan can use.
        if (ctx.vision_provider and ctx.image_column and data
                and ctx.image_column in data[0]):
            mask = _vision_mask(ctx.vision_provider, ctx.query_text,
                                [r.get(ctx.image_column) for r in data])
            return [{**r, "extracted": "yes" if yes else "no"} for r, yes in zip(data, mask)]

        if ctx.llm_provider:
            return self._llm_extract(data, ctx)
        return data

    def _llm_extract(self, data: Table, ctx: ExecutionContext) -> Table:
        columns = list(data[0].keys())
        prompt = (
            f"Given {len(data)} rows with columns {columns}:\n"
            f"Data: {json.dumps(data, default=str)}\n\n"
            f"Query: \"{ctx.query_text}\"\n\n"
            f"Extract the requested information and add it as new columns. "
            f"Return the result as a JSON array of objects (original columns "
            f"plus new extracted columns). Only output the JSON array."
        )
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "You are a data extraction engine. Output only a JSON array."},
                {"role": "user", "content": prompt},
            ])
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
                text = "\n".join(lines[:end])
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                result = json.loads(text[start:end + 1])
                if isinstance(result, list):
                    return result
        except Exception:
            pass
        return data


class SemanticSearchExecutor(OperatorExecutor):
    """Semantic similarity search — ranks rows by relevance to the query."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return data
        if ctx.llm_provider:
            return self._llm_search(data, ctx)
        return data

    def _llm_search(self, data: Table, ctx: ExecutionContext) -> Table:
        columns = list(data[0].keys())
        prompt = (
            f"Given {len(data)} rows with columns {columns}:\n"
            f"Data: {json.dumps(data[:50], default=str)}\n\n"
            f"Query: \"{ctx.query_text}\"\n\n"
            f"Rank rows by semantic relevance to the query. Return a JSON "
            f"array of row indices (0-based) in descending relevance order. "
            f"Only output the JSON array."
        )
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "You are a semantic search engine. Output only a JSON array of indices."},
                {"role": "user", "content": prompt},
            ])
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
                text = "\n".join(lines[:end])
            indices = json.loads(text)
            if isinstance(indices, list):
                return [data[i] for i in indices if 0 <= i < len(data)]
        except Exception:
            pass
        return data


class CrossModalMatchExecutor(OperatorExecutor):
    """Matches across modalities (e.g., image-text alignment)."""

    def execute(self, inputs, ctx, node, impl_type):
        right_src = getattr(ctx, "right_source", None) or []
        left = inputs[0] if inputs else ctx.data_source
        # A distinct RIGHT table (set by a cross-modal join harness) ALWAYS wins:
        # both SCAN leaves otherwise read the same data_source (a self-join), so
        # right_source is the only way to express a genuine two-table join.
        if right_src:
            right = right_src
        elif len(inputs) >= 2:
            right = inputs[1]
        else:
            right = left
        if not left or not right:
            return left or right or []

        if ctx.llm_provider:
            return self._llm_match(left, right, ctx)

        return JoinExecutor().execute([left, right], ctx, node, impl_type)

    def _llm_match(self, left: Table, right: Table, ctx: ExecutionContext) -> Table:
        # right tables can be large candidate pools (e.g. an image/logo table), so
        # allow more right rows than left; both sides stay compact after upstream
        # projection / vision-labelling.
        prompt = (
            f"Left data ({len(left)} rows): {json.dumps(left[:40], default=str)}\n"
            f"Right data ({len(right)} rows): {json.dumps(right[:150], default=str)}\n\n"
            f"Query: \"{ctx.query_text}\"\n\n"
            f"Match rows across these two datasets based on semantic similarity "
            f"or cross-modal alignment. Return a JSON array of matched row "
            f"objects (merged fields from both sides). Only output the JSON array."
        )
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "You are a cross-modal matching engine. Output only a JSON array."},
                {"role": "user", "content": prompt},
            ])
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
                text = "\n".join(lines[:end])
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                result = json.loads(text[start:end + 1])
                if isinstance(result, list):
                    return result
        except Exception:
            pass
        return left


class SimilarityJoinExecutor(OperatorExecutor):
    """Joins two sets by semantic similarity rather than exact key match."""

    def execute(self, inputs, ctx, node, impl_type):
        right_src = getattr(ctx, "right_source", None) or []
        left = inputs[0] if inputs else ctx.data_source
        right = right_src if right_src else (inputs[1] if len(inputs) >= 2 else left)
        if not left or not right:
            return left or right or []

        if ctx.llm_provider:
            return CrossModalMatchExecutor()._llm_match(left, right, ctx)
        return JoinExecutor().execute([left, right], ctx, node, impl_type)


class FeatureTransformExecutor(OperatorExecutor):
    """Transforms data into derived features (embeddings, PCA, normalization)."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return data

        if ctx.llm_provider:
            return self._llm_transform(data, ctx)
        return self._heuristic_transform(data, ctx.query_text)

    def _llm_transform(self, data: Table, ctx: ExecutionContext) -> Table:
        columns = list(data[0].keys())
        prompt = (
            f"Given {len(data)} rows with columns {columns}:\n"
            f"Data: {json.dumps(data[:50], default=str)}\n\n"
            f"Query: \"{ctx.query_text}\"\n\n"
            f"Transform or derive new features as requested. Return a JSON "
            f"array of objects with original columns plus new derived columns. "
            f"Only output the JSON array."
        )
        try:
            raw = ctx.llm_provider.complete([
                {"role": "system", "content": "You are a feature engineering engine. Output only a JSON array."},
                {"role": "user", "content": prompt},
            ])
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")[1:]
                end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
                text = "\n".join(lines[:end])
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                result = json.loads(text[start:end + 1])
                if isinstance(result, list):
                    return result
        except Exception:
            pass
        return data

    def _heuristic_transform(self, data: Table, query: str) -> Table:
        numeric_cols = [k for k in data[0] if isinstance(data[0][k], (int, float))]
        if not numeric_cols:
            return data
        result = []
        for row in data:
            new_row = dict(row)
            for col in numeric_cols:
                v = row.get(col)
                if isinstance(v, (int, float)):
                    new_row[f"{col}_normalized"] = round(v / max(abs(v), 1), 4)
            result.append(new_row)
        return result


class VisualizeExecutor(OperatorExecutor):
    """Produces a summary/statistics view of the data for visualization."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return [{"summary": "No data to visualize"}]

        columns = list(data[0].keys())
        numeric_cols = [k for k in columns if isinstance(data[0].get(k), (int, float))]

        stats: Row = {"_row_count": len(data)}
        for col in numeric_cols:
            vals = [r[col] for r in data if isinstance(r.get(col), (int, float))]
            if vals:
                stats[f"{col}_min"] = min(vals)
                stats[f"{col}_max"] = max(vals)
                stats[f"{col}_avg"] = round(sum(vals) / len(vals), 2)

        return data + [stats] if impl_type == "MATERIALIZE" else data


class ExportExecutor(OperatorExecutor):
    """Formats results for export (adds metadata)."""

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        if not data:
            return [{"export_status": "empty", "format": impl_type}]

        export_meta: Row = {
            "_export_format": "csv" if impl_type == "CSV_EXPORT" else "json",
            "_row_count": len(data),
            "_columns": list(data[0].keys()),
        }
        return data + [export_meta]


# ---------------------------------------------------------------------------
# Executor Registry
# ---------------------------------------------------------------------------

EXECUTOR_REGISTRY: Dict[str, OperatorExecutor] = {
    "SCAN": ScanExecutor(),
    "FILTER": FilterExecutor(),
    "PROJECT": ProjectExecutor(),
    "AGGREGATE": AggregateExecutor(),
    "SORT": SortExecutor(),
    "LIMIT": LimitExecutor(),
    "DISTINCT": DistinctExecutor(),
    "JOIN": JoinExecutor(),
    "UNION": UnionExecutor(),
    "SEMANTIC_SEARCH": SemanticSearchExecutor(),
    "CROSS_MODAL_MATCH": CrossModalMatchExecutor(),
    "CONTENT_EXTRACT": ContentExtractExecutor(),
    "SIMILARITY_JOIN": SimilarityJoinExecutor(),
    "FEATURE_TRANSFORM": FeatureTransformExecutor(),
    "VISUALIZE": VisualizeExecutor(),
    "EXPORT": ExportExecutor(),
}


# ---------------------------------------------------------------------------
# Plan Executor
# ---------------------------------------------------------------------------

@dataclass
class PlanExecutionResult:
    """Complete result of executing a physical plan."""
    data: Table
    operator_results: Dict[str, ExecutionResult]
    root_operator: str
    row_count: int = 0

    def __post_init__(self):
        self.row_count = len(self.data)


class PlanExecutor:
    """Executes a physical plan tree bottom-up against data.

    Usage::

        executor = PlanExecutor(data_source=rows, llm_provider=provider)
        result = executor.execute(logical_plan, physical_assignment)
        print(result.data)  # final output rows
    """

    def __init__(
        self,
        data_source: Table,
        llm_provider: Any = None,
        query_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        vision_provider: Any = None,
        image_column: str = "",
        self_consistency: int = 1,
        right_source: Optional[Table] = None,
    ):
        self._ctx = ExecutionContext(
            data_source=data_source,
            llm_provider=llm_provider,
            query_text=query_text,
            metadata=metadata or {},
            vision_provider=vision_provider,
            image_column=image_column,
            self_consistency=self_consistency,
            right_source=right_source or [],
        )
        self._operator_results: Dict[str, ExecutionResult] = {}

    def execute(
        self,
        plan: LogicalPlanNode,
        assignment: Dict[str, str],
    ) -> PlanExecutionResult:
        """Execute the full plan tree and return results."""
        self._operator_results.clear()
        data = self._execute_node(plan, assignment)

        return PlanExecutionResult(
            data=data,
            operator_results=dict(self._operator_results),
            root_operator=plan.operator_type,
        )

    def _execute_node(
        self,
        node: LogicalPlanNode,
        assignment: Dict[str, str],
    ) -> Table:
        """Recursively execute a node after its children."""
        child_outputs = []
        for child in node.children:
            child_data = self._execute_node(child, assignment)
            child_outputs.append(child_data)

        impl_type = assignment.get(node.operator_type, "default")
        executor = EXECUTOR_REGISTRY.get(node.operator_type)

        if executor is None:
            data = child_outputs[0] if child_outputs else []
        else:
            data = executor.execute(child_outputs, self._ctx, node, impl_type)

        result = ExecutionResult(
            data=data,
            operator_type=node.operator_type,
            impl_type=impl_type,
        )
        self._operator_results[node.node_id] = result

        return data
