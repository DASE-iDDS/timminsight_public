"""Build a MultimodalMetadataGraph (TiMetagraph) from a REAL multimodal dataset,
and emit the grounding context that the TiQC clarifier consumes.

This replaces the previously hardcoded `SCHEMAS` text surrogate in the TiQC pilot
with a metadata graph genuinely CONSTRUCTED from the data: one TABLE modality
carrying the column schema, an IMAGE (and/or TEXT) modality when the table has an
image / long-text column, and CrossModalEntity nodes for the representative values
of low-cardinality categorical columns. `schema_context()` serialises the graph
into the compact schema+values+modality-routing string used to GROUND clarification
(so the `grounded` arm = TiQC + TiMetagraph, `ungrounded` = TiQC alone).

No fabricated data: every column, value and modality here is read from the real
benchmark files.
"""
from __future__ import annotations
import os
import csv
from collections import Counter
from typing import Dict, List, Any, Optional

from .metadata_graph import (
    MultimodalMetadataGraph, ModalityType, EntityType, CrossModalEntity,
    EntityMention, ModalityMetadata,
)

csv.field_size_limit(10 ** 9)

# Real dataset files (relative to repo root), same sources the eval reads.
NIR = "data/nirvana_repo/nirvana-main"
DATASET_FILES = {
    "estate": f"{NIR}/testdata/multimodal_real_estate.parquet",
    "steam":  f"{NIR}/testdata/steam_games.csv",
    "movie":  f"{NIR}/testdata/movie_data.csv",
    "imdb":   f"{NIR}/testdata/movie_data.csv",
}
# Columns whose ground truth lives in IMAGE pixels, not text (per benchmark design).
IMAGE_COLS = {"estate": ["image"], "steam": ["image", "rating"], "movie": ["Poster"]}
# Long free-text columns (TEXT modality).
TEXT_COLS = {
    "estate": ["Details"], "steam": ["description", "overall_reviews", "tags", "genre"],
    "movie": ["Plot"], "imdb": ["Plot"],
}


def _load_rows(dataset: str, sample: int = 100) -> List[Dict[str, Any]]:
    path = DATASET_FILES.get(dataset)
    if not path or not os.path.exists(path):
        return []
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        if sample:
            df = df.head(sample)
        return df.to_dict("records")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:sample] if sample else rows


def build_metadata_graph(dataset: str, sample: int = 100) -> MultimodalMetadataGraph:
    """Construct a MultimodalMetadataGraph for one dataset from its real rows."""
    g = MultimodalMetadataGraph(graph_id=f"meta_{dataset}")
    rows = _load_rows(dataset, sample)
    if not rows:
        return g
    cols = list(rows[0].keys())
    img_cols = [c for c in IMAGE_COLS.get(dataset, []) if c in cols]
    txt_cols = [c for c in TEXT_COLS.get(dataset, []) if c in cols]

    # column -> representative (most common) values, for low-cardinality columns
    reps: Dict[str, List[str]] = {}
    for c in cols:
        if c in img_cols:
            continue
        vals = [str(r.get(c, "")).strip() for r in rows if str(r.get(c, "")).strip()]
        uniq = set(vals)
        if 1 < len(uniq) <= 40 and c not in txt_cols:   # categorical
            reps[c] = [v for v, _ in Counter(vals).most_common(6)]

    # TABLE modality carries the column schema
    g.add_modality(ModalityMetadata(
        modality_id=f"{dataset}_table", modality_type=ModalityType.TABLE,
        source_path=DATASET_FILES.get(dataset, dataset),
        content_summary={"columns": cols, "n_rows": len(rows)},
        semantic_features={"representative_values": reps},
        raw_metadata={"image_columns": img_cols, "text_columns": txt_cols}))
    # IMAGE modality when the table has real image columns
    if img_cols:
        g.add_modality(ModalityMetadata(
            modality_id=f"{dataset}_image", modality_type=ModalityType.IMAGE,
            source_path=DATASET_FILES.get(dataset, dataset),
            content_summary={"image_columns": img_cols},
            raw_metadata={"note": "visual attributes resolvable only from pixels"}))
    # TEXT modality for long free-text columns
    if txt_cols:
        g.add_modality(ModalityMetadata(
            modality_id=f"{dataset}_text", modality_type=ModalityType.TEXT,
            source_path=DATASET_FILES.get(dataset, dataset),
            content_summary={"text_columns": txt_cols}))

    # CrossModalEntity per representative categorical value (grounds value_content
    # / schema_structure ambiguities in the real values that actually occur)
    for c, vals in reps.items():
        for v in vals[:6]:
            eid = f"{dataset}:{c}:{v}"[:120]
            g.add_entity(CrossModalEntity(
                entity_id=eid, entity_type=EntityType.CONCEPT, canonical_name=v,
                mentions=[EntityMention(f"{dataset}_table", ModalityType.TABLE, v,
                                        context=c)],
                properties={"column": c}))
    return g


def schema_context(dataset: str, graph: Optional[MultimodalMetadataGraph] = None,
                   sample: int = 100) -> str:
    """Compact grounding string DERIVED from the metadata graph: columns, per-column
    representative real values, and modality routing (which facts are image-only).
    This is what the TiQC `grounded` arm sees; `ungrounded` sees "" instead."""
    g = graph or build_metadata_graph(dataset, sample)
    tbl = g.modalities.get(f"{dataset}_table")
    if tbl is None:
        return ""
    cols = tbl.content_summary.get("columns", [])
    reps = tbl.semantic_features.get("representative_values", {})
    img_cols = tbl.raw_metadata.get("image_columns", [])
    txt_cols = tbl.raw_metadata.get("text_columns", [])
    parts = [f"Table {dataset}({', '.join(cols)})."]
    if img_cols:
        parts.append(f"REAL image column(s): {', '.join(img_cols)} — visual attributes "
                     f"depicted in these pixels are resolvable ONLY from the image, not "
                     f"from any text column (no such attribute column exists).")
    if txt_cols:
        parts.append(f"Long free-text column(s): {', '.join(txt_cols)}.")
    for c, vals in reps.items():
        parts.append(f"Column '{c}' representative values: {', '.join(map(str, vals))}.")
    return " ".join(parts)


def build_metadata_graph_llm(dataset: str, sample: int = 100, provider=None,
                             vision_provider=None, enable_tools: bool = True):
    """Paper-faithful build (§3.1.1): run the LLM/VLM-driven TiProcessor family under
    ReAct with the real external tools (YOLO/Detectron2/Places365/ViT/spaCy), then merge
    cross-modal entities. Falls back to the fast schema-only build if no provider is given.
    schema_context() works on the returned graph unchanged (same table-modality layout)."""
    if provider is None:
        return build_metadata_graph(dataset, sample)
    from .processors import extract_metadata_graph
    rows = _load_rows(dataset, sample)
    return extract_metadata_graph(
        dataset, rows, provider, vision_provider, enable_tools=enable_tools,
        image_cols=IMAGE_COLS.get(dataset, []), text_cols=TEXT_COLS.get(dataset, []))


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "estate"
    g = build_metadata_graph(ds)
    print(g.get_summary())
    print("\nschema_context:\n", schema_context(ds, g))
