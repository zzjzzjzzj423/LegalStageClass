# LegalStageClass

**Where to Classify? Comparing Classification Before, During, and After Legal Retrieval**

LegalStageClass is a small, runnable experiment for Chinese legal retrieval on STARD. It compares a shared hybrid baseline with two independent Jev interventions before and after retrieval.

## Research question

How do independent Jev interventions before and after hybrid legal retrieval affect retrieval quality, latency, and API cost?

## Compared strategies

1. **`baseline_hybrid`**
   Query → dynamic Top-4-domain directory (at most 255 files) → local title-similarity Top-2 files → BM25/local BGE Dense Top-1000 → RRF → Top-10 articles.

2. **`jev_pre_route` - file Choice**
   On the exact same domain-limited file directory, Jev Choice ranks all file IDs and titles; its top two files are then retrieved exactly as in the baseline.

3. **`jev_post_rerank` - article Choice**
   It reuses the baseline file selection and complete RRF ranking, sends its Top-20 articles to Jev Choice, and uses probabilities to form the final Top-10.

Each strategy is evaluated independently on top of the same hybrid baseline rather than being stacked into one system.

## Baselines and controls

- BM25 + SiliconFlow BGE-M3 dense retrieval with reciprocal rank fusion
- PRE and POST are separate arms and are never stacked

## Evaluation

Retrieval quality:

- Recall@10
- nDCG@10
- Hit@1
- Domain and file-stage gold retention
- Shortlist recall and actual shortlist size for POST

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

`data/raw/`, generated datasets under `data/processed/`, and `artifacts/` are ignored by Git. The two generated domain-classification deliverables documented in `data/README.md` are explicitly tracked exceptions; their processing code lives under `data/processed/`. Do not add API keys, downloaded data, or run results to a commit.

Install the package and derive the normalized dataset:

```bash
python -m pip install -e .
legal-stage-class data prepare --raw-root data/raw/STARD
```

The three experiment arms also require a local `BAAI/bge-small-zh-v1.5`
checkpoint for domain and title navigation. Install its optional runtime and
place the checkpoint in the local Hugging Face cache (the default is offline):

```bash
python -m pip install -e '.[navigation]'
```

The command writes only these normalized files under `data/processed/stard/<dataset_hash>/`: `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl`, `splits.json`, and a mechanically derived `law_catalog.jsonl`. The catalog groups article names by the final `第X条` or `第X条之X` suffix.

For the current upstream snapshot, preparation produces:

```text
data/processed/stard/d2bc0bf33155184c150c400cf1b29e03230e0dfc5c0a098dac5fdb3a59020482/
```

The default Dense backend is the same local `BAAI/bge-small-zh-v1.5` model, so only PRE/POST need `TYPESAFE_API_KEY`. To reproduce the former remote BGE-M3 variant, add `--dense-backend siliconflow` and set `SILICONFLOW_API_KEY`:

```bash
export TYPESAFE_API_KEY='your-key'
```

The first run embeds all 55,348 corpus articles locally and reports progress. This is a one-time local computation; its vectors are cached for later arms and runs.

Corpus embeddings are stored locally under `.cache/legal-stage-class/` after the first successful build. The cache is ignored by Git and is never shared through the repository. A later `run` or `acceptance` command reuses it only when the dataset hash, full corpus text, SiliconFlow endpoint, model, encoding format, and input template all match. The three arms in one `acceptance` process also share the same in-memory vectors.

Jev decisions use the same local cache root, under `.cache/legal-stage-class/jev/`. PRE caches probabilities for its exact domain-limited file directory; POST caches probabilities for its exact article shortlist. A cached decision is reused only if the query, candidate text, Jev endpoint/model, and prompt contract are unchanged. The cache holds provider outputs locally: do not commit or copy it into Git.

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

For the budget-bounded 600-query experiment, use the committed, stratified `tune_train` selection. It excludes the 309-query `final` split; its exact query IDs, seed, strata, and integrity hash are in `configs/stard_tune_train_600_v1.json`. On macOS, pass the list to the CLI with:

```bash
QUERY_IDS="$(jq -r '.query_ids | join(",")' configs/stard_tune_train_600_v1.json)"
```

For the first live acceptance, use exactly three fixed query IDs. This runs baseline, then PRE, then POST:

```bash
legal-stage-class acceptance \
  --dataset data/processed/stard/<dataset_hash> \
  --query-ids 11,12,13
```

Each arm writes only `manifest.json`, `rankings.jsonl`, `metrics.json`, `latency_cost.jsonl`, and `provider_ledger.jsonl` to `artifacts/runs/<run_id>/`. Set provider cost variables before a paid run so the ledger uses your current pricing.

## Implementation contract

The following behavior is shared across collaborators and should not change silently:

| Arm | Fixed behavior | Arm-specific behavior |
| --- | --- | --- |
| `baseline_hybrid` | Dynamic Top-4 domain directory, title-similarity Top-2 files, BM25 + local BGE Top-1000, RRF (`k=60`) | None |
| `jev_pre_route` | Same domain directory, 255-file cap, file budget, retrieval, metrics, and corpus vectors | Jev Choice ranks the whole file directory by ID and title; probability Top-2 replaces title similarity |
| `jev_post_rerank` | Same domain directory, title-selected files, and complete RRF ranking as baseline | Jev Choice reranks RRF Top-20 article ID/name/content options; probability order yields final Top-10 |

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
3. Use `tune_train` to confirm the Jev contracts and choose `post_shortlist_n`; do not use the reserved final split for that decision.
4. Record live Jev input tokens, latency, and cost on a bounded development run before evaluating more queries.
5. Expand to wider `tune_train` evaluation only after the cache and small live smoke check behave as expected.

Do not commit `data/`, `artifacts/`, `.cache/`, API keys, or provider payloads. Commit source code, tests, configuration semantics, and documentation changes together so collaborators can reproduce the same behavior locally.

## Data

The project uses STARD: *A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals* (Findings of EMNLP 2024).

- Paper: <https://aclanthology.org/2024.findings-emnlp.625/>

