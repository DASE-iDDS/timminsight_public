"""
Natural Language Answer Generator.

Takes execution results and the original query, produces a human-readable
natural language answer. Uses LLM when available; falls back to template-based
generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .plan_executor import PlanExecutionResult, Table


def _clean_for_answer(data: Table) -> Table:
    """Drop raw image bytes and image columns and shorten long text so the NL
    answer (and the LLM answer prompt) is not polluted by base64/JPEG bytes
    (which otherwise made the answer unjudgeable garbage on image datasets)."""
    out = []
    for row in data:
        clean = {}
        for k, v in row.items():
            kl = str(k).lower()
            if isinstance(v, (bytes, bytearray)) or "image" in kl or kl in ("poster", "rating", "url"):
                continue
            if isinstance(v, str):
                if v.startswith("data:image"):
                    continue
                if len(v) > 200:
                    v = v[:200] + "..."
            clean[k] = v
        out.append(clean)
    return out


class AnswerGenerator:
    """Generates natural language answers from execution results.

    Usage::

        gen = AnswerGenerator(llm_provider=provider)
        answer = gen.generate(query_text, execution_result)
    """

    def __init__(self, llm_provider: Any = None):
        self._llm = llm_provider

    def generate(
        self,
        query_text: str,
        execution_result: PlanExecutionResult,
    ) -> str:
        """Generate a natural language answer."""
        if self._llm:
            return self._llm_generate(query_text, execution_result)
        return self._template_generate(query_text, execution_result)

    def _llm_generate(
        self,
        query_text: str,
        result: PlanExecutionResult,
    ) -> str:
        data = _clean_for_answer(result.data)
        data_preview = data[:20] if len(data) > 20 else data

        prompt = (
            f"User's question: \"{query_text}\"\n\n"
            f"Query execution returned {result.row_count} rows.\n"
            f"Data: {json.dumps(data_preview, default=str)}\n\n"
            f"Based on this data, provide a clear, concise natural language "
            f"answer to the user's question. If the data contains aggregate "
            f"results (counts, averages, etc.), state the numbers directly. "
            f"If it contains filtered rows, summarize what was found. "
            f"Be specific with numbers and key findings."
        )
        try:
            raw = self._llm.complete([
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst assistant. Given query results, "
                        "produce a clear natural language answer. Be concise and precise."
                    ),
                },
                {"role": "user", "content": prompt},
            ])
            return raw.strip()
        except Exception:
            return self._template_generate(query_text, result)

    def _template_generate(
        self,
        query_text: str,
        result: PlanExecutionResult,
    ) -> str:
        data = _clean_for_answer(result.data)

        if not data:
            return f"No results found for: {query_text}"

        if len(data) == 1:
            row = data[0]
            if "count" in row:
                return f"Count: {row['count']}"
            if "average" in row:
                return f"Average {row.get('column', 'value')}: {row['average']}"
            if "sum" in row:
                return f"Total {row.get('column', 'value')}: {row['sum']}"
            if "min" in row:
                return f"Minimum {row.get('column', 'value')}: {row['min']}"
            if "max" in row:
                return f"Maximum {row.get('column', 'value')}: {row['max']}"
            if "result" in row:
                return f"Result: {row['result']}"
            parts = [f"{k}: {v}" for k, v in row.items()]
            return "; ".join(parts)

        columns = list(data[0].keys())
        lines = [f"Found {len(data)} results:"]

        for i, row in enumerate(data[:10]):
            vals = [f"{k}={row.get(k)}" for k in columns[:4]]
            lines.append(f"  {i+1}. {', '.join(vals)}")

        if len(data) > 10:
            lines.append(f"  ... and {len(data) - 10} more rows")

        return "\n".join(lines)
