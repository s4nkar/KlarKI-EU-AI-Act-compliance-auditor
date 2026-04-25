# KlarKI Documentation

**KlarKI** is a local-first EU AI Act + GDPR compliance auditor. All inference runs on-device — no external API calls, no data leaves the machine.

## Documents

| File | What it covers |
|---|---|
| [architecture.md](architecture.md) | Service topology, containers, data flow, storage |
| [training-pipeline.md](training-pipeline.md) | Full training walkthrough: data generation → BERT → NER → specialist classifiers |
| [inference-pipeline.md](inference-pipeline.md) | Full inference walkthrough: upload → chunking → classification → RAG → LangGraph → report |
| [rag.md](rag.md) | RAG system deep-dive: BM25, vector search, RRF, cross-encoder, metadata filtering |
| [models.md](models.md) | Every ML model: what it does, training targets, where it runs in inference |
| [configuration.md](configuration.md) | Every knob you can turn: training data size, Triton, OpenSearch, Ollama model, etc. |

## Quick start

```bash
cp .env.example .env
./run.sh setup          # first time — builds everything (~30–60 min)
./run.sh up             # day-to-day (production, nginx)
./run.sh dev            # development (hot reload, port 3000)
./run.sh triton         # GPU inference via Triton (NVIDIA GPU required)
```

## Key design decisions

- **Local-first**: no external API calls; Ollama runs the LLM on-device
- **Deterministic**: `temperature=0, seed=42` on all LLM calls — reproducible results
- **Applicability-gated**: non-applicable articles are never sent to the LLM — saves latency
- **Dual backends**: Ollama (CPU-friendly) or Triton (GPU, ~50–100× faster for BERT)
- **Bilingual**: EN/DE documents supported throughout
