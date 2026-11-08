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

### Dataset setup for collaborators

The dataset is intentionally not committed to this repository. Each collaborator downloads the same two public STARD source files locally, then derives the identical processed dataset. From the repository root:

```bash
mkdir -p data/raw/STARD
curl --fail --location --output data/raw/STARD/corpus.jsonl \
  https://raw.githubusercontent.com/oneal2000/STARD/main/data/corpus.jsonl
curl --fail --location --output data/raw/STARD/queries.json \
  https://raw.githubusercontent.com/oneal2000/STARD/main/data/queries.json
```

`data/raw/`, `data/processed/`, and `artifacts/` are ignored by Git. Do not add API keys, downloaded data, or run results to a commit.

Install the package and derive the normalized dataset:

```bash
python -m pip install -e .
legal-stage-class data prepare --raw-root data/raw/STARD
```

The command writes only these normalized files under `data/processed/stard/<dataset_hash>/`: `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl`, `splits.json`, and a mechanically derived `law_catalog.jsonl`. The catalog groups article names by the final `第X条` or `第X条之X` suffix.

For the current upstream snapshot, preparation produces:

```text
data/processed/stard/d2bc0bf33155184c150c400cf1b29e03230e0dfc5c0a098dac5fdb3a59020482/
```

Set `SILICONFLOW_API_KEY` for all arms and `TYPESAFE_API_KEY` for PRE/POST in the shell that runs the experiment. Dense retrieval always calls SiliconFlow `BAAI/bge-m3`; it does not fall back to local embeddings:

```bash
export SILICONFLOW_API_KEY='your-key'
export TYPESAFE_API_KEY='your-key'
```

The first live run embeds all 55,348 corpus articles. It reports batch progress and uses four concurrent SiliconFlow embedding requests by default. If the provider returns HTTP `429`, lower it before retrying:

```bash
export SILICONFLOW_EMBEDDING_CONCURRENCY=1
```

Corpus embeddings are stored locally under `.cache/legal-stage-class/` after the first successful build. The cache is ignored by Git and is never shared through the repository. A later `run` or `acceptance` command reuses it only when the dataset hash, full corpus text, SiliconFlow endpoint, model, encoding format, and input template all match. The three arms in one `acceptance` process also share the same in-memory vectors.

Use a different local location if needed:

```bash
legal-stage-class acceptance --cache-root /path/to/local/cache ...
```

Run one arm on a bounded query set with:

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

## Implementation contract

The following behavior is shared across collaborators and should not change silently:

| Arm | Fixed behavior | Arm-specific behavior |
| --- | --- | --- |
| `baseline_hybrid` | BM25 Top-1000 + SiliconFlow `BAAI/bge-m3` Top-1000 → RRF (`k=60`) → Top-10 | None |
| `jev_pre_route` | Same BM25, Dense, RRF, Top-10, metrics, and corpus vectors as baseline | Jev Noul scores all laws; `threshold`, `min_laws`, and `max_laws` select an article allow-list before retrieval |
| `jev_post_rerank` | Same complete baseline Top-10 and common metrics | Jev Choice receives only that Top-10; option IDs are always `a_<article_id>` and probability order becomes the final ranking |

PRE and POST are independent experimental arms. Never enable or combine them in one run. Dense retrieval must remain SiliconFlow `BAAI/bge-m3`; do not silently substitute a local embedding model.

## Run-artifact contract

| File | What it contains | Development use |
| --- | --- | --- |
| `manifest.json` | Dataset hash, query IDs, parameters, provider-cost summary, cache source/key | Confirm two runs used comparable inputs |
| `rankings.jsonl` | One row per query: shortlist, final Top-10, ranks, PRE route or POST decision metadata | Recompute or inspect ranking changes |
| `metrics.json` | Macro retrieval metrics and per-query values | Compare arms only on the same query set |
| `latency_cost.jsonl` | Per-query `query_to_final_topk_ms`, optional POST increment, reserved/estimated cost | Separate online latency from corpus-index build time |
| `provider_ledger.jsonl` | One row per remote provider request, usage, latency, request ID, cost status | Audit API volume and cost estimates |

`cost_usd` is an estimate or reservation unless the provider supplies an authoritative billed amount. A value of `0` for SiliconFlow reflects the configured reservation, not proof that the provider charged nothing.

## Development guide

1. Keep the existing 3-query acceptance command as the first live smoke check after changes to retrieval, providers, metrics, or artifact writing.
2. Add focused offline tests under `tests/` for every new parser, ranker, provider-response rule, cache identity change, or metric edge case. Run `PYTHONPATH=src python -m pytest` before committing.
3. Use `tune_train` for threshold, route-breadth, or cost calibration. Do not use a reserved final split to choose those parameters.
4. Before a run larger than ten PRE queries, record the ten-query cost calibration and stop when the projected cost exceeds the authorized budget. With the default reservation of `$0.00003` per law judgment, all 1,543 STARD queries project to about `$58.23`, above the default `$20` ceiling.
5. Expand to wider `tune_train` baseline/POST evaluation only after the cache and 10-query PRE calibration behave as expected. Consider a final evaluation or additional research controls only after those results are reviewed.

Do not commit `data/`, `artifacts/`, `.cache/`, API keys, or provider payloads. Commit source code, tests, configuration semantics, and documentation changes together so collaborators can reproduce the same behavior locally.

## Data

The project uses STARD: *A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals* (Findings of EMNLP 2024).

- Paper: <https://aclanthology.org/2024.findings-emnlp.625/>

