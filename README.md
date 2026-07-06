# TiMMInsight

Automated multimodal exploratory data analysis (EDA) with large language models. The
system turns a natural-language question over a multimodal dataset (relational tables,
text, and images) into an accurate answer, through three components:

- **TiMetaGraph** — builds a metadata graph from multimodal data using a family of
  LLM-driven processors (`TiProcessor`) under the ReAct paradigm, calling dedicated
  external tools (object detection, scene classification, semantic features, NER,
  sentiment, topic extraction), and discovers cross-modal entity relationships.
- **TiQC** — an interactive query-clarification module that detects and classifies query
  ambiguity into a six-type taxonomy and resolves it through a metadata-grounded,
  multi-round human-in-the-loop dialogue.
- **TiTSP** — an uncertainty-aware two-stage query planner: a Monte Carlo Tree Search over
  logical plans constrained by an operator compatibility graph, with bottom-up uncertainty
  propagation, followed by multi-objective (Pareto) physical-plan optimization with a
  weighted Tchebycheff objective.

## Requirements

Python 3.12+. Install dependencies with `pip install -r requirements.txt`, then download the
spaCy model:

```
python -m spacy download en_core_web_sm
```

LLM access is configured via environment variables (an OpenAI-compatible endpoint / API key).
