"""
LLM-based Query Analyzer for Multimodal Query Planning.

Replaces regex-based keyword matching with LLM inference to extract
operator requirements from natural language queries. Supports multiple
LLM providers via a unified OpenAI-compatible chat completions interface.

Supported providers:
  - DashScope (Qwen, Kimi, etc.)
  - OpenAI (GPT-4o, GPT-5, etc.)
  - AWS Bedrock (Claude, etc.)
  - Any OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .operator_compatibility_graph import (
    ModalityType,
    OperatorCompatibilityGraph,
    QueryContext,
)

# ---------------------------------------------------------------------------
# Operator Description Registry
# ---------------------------------------------------------------------------

OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    "SCAN": "Read raw data from a table or collection (always required as leaf)",
    "FILTER": "Apply selection predicates (WHERE conditions, thresholds, semantic filtering)",
    "PROJECT": "Select specific columns/fields",
    "AGGREGATE": "Group-by and aggregate (COUNT, SUM, AVG, etc.)",
    "SORT": "Order results by one or more keys",
    "LIMIT": "Restrict output to top-N rows",
    "DISTINCT": "Remove duplicate rows",
    "JOIN": "Combine two data sources on a key",
    "UNION": "Merge multiple result sets",
    "SEMANTIC_SEARCH": "Vector similarity search using embeddings (semantic meaning, content matching)",
    "CROSS_MODAL_MATCH": "Match across modalities (e.g., image-text alignment, captioning)",
    "CONTENT_EXTRACT": "Extract structured info from unstructured data (object detection, OCR, NER, semantic mapping)",
    "SIMILARITY_JOIN": "Join two sets by embedding similarity (nearest neighbors)",
    "FEATURE_TRANSFORM": "Convert data to embeddings (vectorize, PCA, t-SNE)",
    "VISUALIZE": "Produce charts, plots, or visual output",
    "EXPORT": "Save or download results to a file",
}


def build_prompt_from_ocg(
    ocg: Optional[OperatorCompatibilityGraph] = None,
    custom_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Dynamically build the system prompt from OCG operator specs.

    If an OCG is provided, the operator list is generated from its registered
    specs. Otherwise falls back to a default OCG. Custom descriptions can
    override or extend the built-in ones.
    """
    if ocg is None:
        ocg = OperatorCompatibilityGraph()

    descriptions = dict(OPERATOR_DESCRIPTIONS)
    if custom_descriptions:
        descriptions.update(custom_descriptions)

    categories: Dict[str, List[str]] = {}
    category_labels = {
        "access": "Data Access",
        "transform": "Transforms",
        "cross_modal": "Cross-Modal",
        "output": "Output",
    }

    for op_type in ocg.all_operator_types:
        spec = ocg.get_spec(op_type)
        if spec is None:
            continue
        cat = spec.category
        categories.setdefault(cat, []).append(op_type)

    operator_section_lines = []
    for cat_key in ("access", "transform", "cross_modal", "output"):
        ops = categories.get(cat_key, [])
        if not ops:
            continue
        label = category_labels.get(cat_key, cat_key)
        operator_section_lines.append(f"**{label}:**")
        for op in ops:
            desc = descriptions.get(op, op)
            operator_section_lines.append(f"- {op}: {desc}")
        operator_section_lines.append("")

    operator_section = "\n".join(operator_section_lines).strip()
    num_ops = len(ocg.all_operator_types)

    leaf_ops = [
        t for t in ocg.all_operator_types
        if (s := ocg.get_spec(t)) and s.is_leaf
    ]
    terminal_ops = [
        t for t in ocg.all_operator_types
        if (s := ocg.get_spec(t)) and s.is_terminal
    ]

    root_rules = []
    for t in terminal_ops:
        root_rules.append(f"   - If the query asks to {t.lower()} → root = {t}")
    root_rules_str = "\n".join(root_rules)

    leaf_str = ", ".join(leaf_ops) if leaf_ops else "SCAN"

    modalities = ", ".join(f'"{m.value}"' for m in ModalityType)

    return f"""\
You are a multimodal query planner. Given a natural language query about \
a multimodal dataset (tables, images, text), determine which logical \
operators are needed and how they should be structured.

## Available Operators ({num_ops} types)

{operator_section}

## Rules

1. {leaf_str} is always required (leaf operator for data access).
2. The root operator is the topmost operator in the plan tree:
{root_rules_str}
   - Otherwise → root = {leaf_str} (simple retrieval)
3. Required operators MUST appear in the plan. Optional operators MAY improve it.
4. Identify target modalities from the query content.

## Output Format

Return a JSON object (no markdown fences, no extra text):
{{
  "required_operators": ["{leaf_str}", ...],
  "optional_operators": [...],
  "root_operator_type": "...",
  "target_modalities": ["tabular", ...],
  "estimated_data_size": 1000
}}

Valid modalities: {modalities}
"""


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Cumulative LLM token usage across calls on a single provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        # Fall back to prompt+completion if the API omits a total.
        self.total_tokens += int(total or (int(prompt or 0) + int(completion or 0)))
        self.calls += 1

    def reset(self) -> None:
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        self.calls = 0

    def snapshot(self) -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens, self.completion_tokens,
            self.total_tokens, self.calls,
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Every provider exposes a ``usage`` counter (``TokenUsage``) that
    accumulates token consumption across ``complete`` calls, so experiment
    harnesses can snapshot/reset it around a query.
    """

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages and return the assistant's response text."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible chat completions API.

    Works with: OpenAI, DashScope, vLLM, Ollama, LiteLLM, etc.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 75,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._extra_body = extra_body or {}
        self.usage = TokenUsage()

    @property
    def provider_name(self) -> str:
        return f"openai-compatible ({self._base_url})"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        # Provider-specific extras, e.g. {"enable_thinking": False} to turn off
        # the reasoning mode of thinking models such as Qwen3 on DashScope.
        payload.update(self._extra_body)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        import time
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as _FutureTimeout
        last_exc = None

        def _do_request():
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))

        # Hard wall-clock cap per attempt. urllib's own ``timeout`` is silently
        # bypassed under a SOCKS proxy (the socket can freeze uninterruptibly — seen
        # with some endpoints under concurrent large-prompt load), and signal.alarm
        # cannot reach a worker thread, so we run the request in a thread and abandon
        # it if it overruns. A stalled call then fails fast (and retries / falls back)
        # instead of hanging the entire run.
        hard = 100
        timeouts = 0
        for attempt in range(3):
            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(_do_request)
            try:
                body = fut.result(timeout=hard)
                ex.shutdown(wait=False)
                usage = body.get("usage") or {}
                self.usage.add(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )
                return body["choices"][0]["message"]["content"]
            except _FutureTimeout:
                ex.shutdown(wait=False)  # abandon the frozen request thread
                last_exc = TimeoutError(f"request stalled >{hard}s")
                # a stalled connection rarely recovers on retry, so fail fast to the
                # caller's fallback after a single retry instead of burning 3x hard.
                timeouts += 1
                if timeouts >= 2:
                    break
                continue
            except urllib.error.HTTPError as exc:
                ex.shutdown(wait=False)
                last_exc = exc
                # retry rate-limit / transient server errors with exponential backoff
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                    continue
                raise RuntimeError(f"LLM request failed: HTTP {exc.code}") from exc
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                ex.shutdown(wait=False)
                last_exc = exc
                time.sleep(min(20.0, 1.5 * (2 ** attempt)))
                continue
            except KeyError as exc:
                ex.shutdown(wait=False)
                raise RuntimeError(f"LLM request failed: {exc}") from exc
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")


class BedrockProvider(LLMProvider):
    """Provider for AWS Bedrock (uses boto3 converse API)."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-sonnet-4-20250514",
        region: str = "us-east-1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._model_id = model_id
        self._region = region
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.usage = TokenUsage()

    @property
    def provider_name(self) -> str:
        return f"bedrock ({self._model_id})"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        try:
            import boto3
        except ImportError:
            raise RuntimeError("boto3 is required for BedrockProvider: pip install boto3")

        client = boto3.client("bedrock-runtime", region_name=self._region)

        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]

        bedrock_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in user_msgs
        ]

        kwargs: Dict[str, Any] = {
            "modelId": self._model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": self._temperature,
                "maxTokens": self._max_tokens,
            },
        }
        if system_msgs:
            kwargs["system"] = [{"text": system_msgs[0]["content"]}]

        try:
            response = client.converse(**kwargs)
            u = response.get("usage") or {}
            self.usage.add(
                u.get("inputTokens", 0),
                u.get("outputTokens", 0),
                u.get("totalTokens", 0),
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as exc:
            raise RuntimeError(f"Bedrock request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
}


def create_provider(
    provider_type: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Create an LLM provider by type.

    Args:
        provider_type: "dashscope", "openai", "deepseek", "bedrock",
                       or "custom" for arbitrary OpenAI-compatible endpoints.
        model: Model name (e.g., "qwen-plus", "gpt-4o", "kimi-k2").
        api_key: API key. If None, reads from environment.
        base_url: Override base URL. If None, uses preset.
    """
    if provider_type == "bedrock":
        return BedrockProvider(model_id=model, **kwargs)

    preset = PROVIDER_PRESETS.get(provider_type, {})
    resolved_url = base_url or preset.get("base_url")
    if not resolved_url:
        raise ValueError(
            f"Unknown provider '{provider_type}'. "
            f"Use one of: {list(PROVIDER_PRESETS.keys())}, 'bedrock', or 'custom' with base_url."
        )

    resolved_key = api_key or os.environ.get(preset.get("env_key", ""), "")
    if not resolved_key:
        raise ValueError(
            f"API key not found. Set {preset.get('env_key', 'API_KEY')} "
            f"environment variable or pass api_key directly."
        )

    # Disable the reasoning mode of thinking models by default on DashScope
    # (covers Qwen3 and Kimi). Callers may override via extra_body=...
    extra_body = kwargs.pop("extra_body", None)
    if extra_body is None and provider_type == "dashscope":
        extra_body = {"enable_thinking": False}

    return OpenAICompatibleProvider(
        api_key=resolved_key,
        model=model,
        base_url=resolved_url,
        extra_body=extra_body,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LLM Query Analyzer
# ---------------------------------------------------------------------------

class LLMQueryAnalyzer:
    """Analyzes natural language queries using an LLM to extract operator requirements."""

    def __init__(
        self,
        provider: LLMProvider,
        ocg: Optional[OperatorCompatibilityGraph] = None,
        custom_descriptions: Optional[Dict[str, str]] = None,
        metadata_context: Optional[str] = None,
    ):
        self._provider = provider
        self._system_prompt = build_prompt_from_ocg(ocg, custom_descriptions)
        if metadata_context:
            # Ground operator extraction in the metadata graph: knowing which
            # modality holds an attribute decides whether cross-modal
            # operators (CONTENT_EXTRACT, CROSS_MODAL_MATCH, ...) are required
            # or a plain relational FILTER suffices.
            self._system_prompt += (
                "\n\nDataset context (from the metadata graph):\n"
                f"{metadata_context}\n"
                "Use this context to decide which operators are required: an "
                "attribute that resides in images or free text (rather than a "
                "table column) requires the corresponding cross-modal "
                "operator, whereas table-resident attributes need only "
                "relational operators."
            )

    def analyze(self, query_text: str) -> Dict[str, Any]:
        """Send the query to the LLM and parse the structured response."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query_text},
        ]
        raw = self._provider.complete(messages)
        return self._parse_response(raw)

    def to_query_context(self, query_text: str) -> QueryContext:
        """Analyze a query and return a QueryContext directly."""
        result = self.analyze(query_text)

        required = set(result.get("required_operators", ["SCAN"]))
        required.add("SCAN")
        optional = set(result.get("optional_operators", []))
        optional -= required

        root = result.get("root_operator_type", "SCAN")
        if root not in required:
            required.add(root)

        if root == "SCAN" and len(required) > 1:
            root_priority = [
                "VISUALIZE", "EXPORT", "AGGREGATE", "SORT", "DISTINCT",
                "LIMIT", "FILTER", "PROJECT", "CONTENT_EXTRACT",
                "CROSS_MODAL_MATCH", "SEMANTIC_SEARCH", "FEATURE_TRANSFORM",
                "JOIN", "SIMILARITY_JOIN", "UNION",
            ]
            for candidate in root_priority:
                if candidate in required:
                    root = candidate
                    break

        modality_map = {m.value: m for m in ModalityType}
        modalities = set()
        for m in result.get("target_modalities", ["tabular"]):
            if m in modality_map:
                modalities.add(modality_map[m])
        if not modalities:
            modalities.add(ModalityType.TABULAR)

        return QueryContext(
            query_text=query_text,
            required_operators=required,
            optional_operators=optional,
            root_operator_type=root,
            target_modalities=modalities,
            estimated_data_size=result.get("estimated_data_size", 1000),
        )

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Extract JSON from the LLM response, tolerating markdown fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # skip ```json
            end = next((i for i, l in enumerate(lines) if l.strip() == "```"), len(lines))
            text = "\n".join(lines[:end])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
            raise ValueError(f"Could not parse LLM response as JSON: {raw[:200]}")
