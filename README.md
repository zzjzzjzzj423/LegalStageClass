# LegalStageClass

**Where to Classify? Comparing Classification Before, During, and After Legal Retrieval**

LegalStageClass is a small, runnable experiment for Chinese legal retrieval on STARD. It compares a shared hybrid baseline with two independent Jev interventions before and after retrieval.

## Research question

How do independent Jev interventions before and after hybrid legal retrieval affect retrieval quality, latency, and API cost?

## Compared strategies

1. **`baseline_hybrid`**
   STARD query → BM25 Top-1000 + SiliconFlow BGE-M3 Dense Top-1000 → RRF → Top-10 articles.

2. **`jev_pre_route` - multi-law routing**
   Jev selects one or more relevant laws before BM25 and dense retrieval search the corresponding article collections.

3. **`jev_post_rerank` - shortlist selection**
   Jev receives the hybrid top-n shortlist, chooses the most relevant article, and uses option probabilities to rerank the shortlist.

Each strategy is evaluated independently on top of the same hybrid baseline rather than being stacked into one system.

## Baselines and controls

- BM25 + SiliconFlow BGE-M3 dense retrieval with reciprocal rank fusion
- PRE and POST are separate arms and are never stacked

## Evaluation

Retrieval quality:

- Recall@10
- nDCG@10
- Hit@1
- Routing retention for PRE
- Shortlist Recall@10 for POST

Efficiency:

- Per-query latency
- Provider usage and actual/reserved cost

## Repository layout

```text
configs/                 Experiment configuration notes
data/                    Dataset setup documentation (data is not committed)
docs/pipeline.md         Research pipeline and experimental design
proposal/                Original project proposal
src/legal_stage_class/   Implementation package
tests/                   Tests
```

## Minimal runnable experiment

The implementation currently contains only three independent arms: `baseline_hybrid`, `jev_pre_route`, and `jev_post_rerank`. It does not implement concept tagging, a cross-encoder, validation/final locking, or an offline embedding model.

Prepare a downloaded STARD directory containing `corpus.jsonl` and `queries.json` (and, when available, `qrels.train.tsv` / `qrels.dev.tsv`):

```bash
python -m pip install -e .
legal-stage-class data prepare --raw-root data/raw/STARD
```

The command writes only these normalized files under `data/processed/stard/<dataset_hash>/`: `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl`, `splits.json`, and a mechanically derived `law_catalog.jsonl`. The catalog groups article names by the final `第X条` or `第X条之X` suffix.

Set `SILICONFLOW_API_KEY` for all arms and `TYPESAFE_API_KEY` for PRE/POST. Dense retrieval always calls SiliconFlow `BAAI/bge-m3`; it does not fall back to local embeddings. Run one arm on a bounded query set with:

```bash
legal-stage-class run \
  --dataset data/processed/stard/<dataset_hash> \
  --strategy baseline_hybrid \
  --query-ids 11,12,13
```

For the first live acceptance, use exactly three fixed query IDs. This runs baseline, then PRE, then POST:

```bash
legal-stage-class acceptance \
  --dataset data/processed/stard/<dataset_hash> \
  --query-ids 11,12,13
```

Each arm writes only `manifest.json`, `rankings.jsonl`, `metrics.json`, `latency_cost.jsonl`, and `provider_ledger.jsonl` to `artifacts/runs/<run_id>/`. PRE rejects a run of more than ten queries when its ten-query calibration projects Jev reserved/estimated cost above the default `$20` ceiling. Set `TYPESAFE_JEV_RESERVED_USD_PER_JUDGMENT` or a provider token price variable before a paid run so the ledger uses your current pricing.

## Data

The project uses STARD: *A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals* (Findings of EMNLP 2024).

- Paper: <https://aclanthology.org/2024.findings-emnlp.625/>

