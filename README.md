# LegalStageClass

**Where to Classify? Comparing Classification Before, During, and After Legal Retrieval**

LegalStageClass is a small, runnable experiment for Chinese legal retrieval on STARD. It compares a shared hybrid baseline with independent pre-retrieval, post-retrieval, and cross-encoder reranking interventions.

## Research question

How do independent interventions before and after hybrid legal retrieval affect retrieval quality, latency, and API cost?

## Compared strategies

All four strategies build the same file directory from the four highest-ranked domains. They deduplicate by file ID and keep higher-ranked domains first. If a domain would exceed the 255-file budget, local BGE ranks only that domain's new file titles against the query and fills the remaining slots.

1. **`baseline_hybrid`**
   Rank all articles in the directory with BM25 and local BGE Dense, fuse the complete rankings with RRF, and return the top ten.

2. **`jev_pre_route` - file Choice**
   Jev Choice ranks the directory's file IDs and titles. Rank all articles in its top two files with BM25 and Dense, fuse the complete rankings with RRF, and return the top ten.

3. **`jev_post_rerank` - article Choice**
   Use the same RRF ranking as the baseline, send its top 30 articles to Jev Choice, and return the ten highest-probability articles.

4. **`cross_encoder_rerank` - Qwen3-Reranker-8B**
   Use the same RRF ranking as the baseline, send its top 30 articles to SiliconFlow's `Qwen/Qwen3-Reranker-8B`, and return the ten highest-scoring articles.

Each strategy is evaluated independently on top of the same hybrid baseline rather than being stacked into one system.

## Baselines and controls

- BM25 + local BGE dense retrieval with reciprocal rank fusion by default
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

The implementation contains four independent arms: `baseline_hybrid`, `jev_pre_route`, `jev_post_rerank`, and `cross_encoder_rerank`. The cross-encoder arm uses SiliconFlow's rerank API and defaults to `Qwen/Qwen3-Reranker-8B`. It does not implement concept tagging, validation/final locking, or an offline embedding model.

The first 308-query `cross_encoder_rerank` run completed with Recall@10 52.25%, nDCG@10 47.13%, Hit@1 44.16%, and Top-30 shortlist recall 53.11%. This run used local `BAAI/bge-small-zh-v1.5` for initial navigation and Dense retrieval; see [the detailed result record](docs/experiment-results.md#qwen3-reranker-8b-独立后置重排2026-11-10) for the caveat on comparing it with the BGE-M3 + Jev results.

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

The retrieval arms require a local `BAAI/bge-small-zh-v1.5`
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

Run `run` and `acceptance` from the repository root so the CLI reads its Git-ignored `.env` file. Add your Jev key there for PRE/POST and your SiliconFlow key for the cross-encoder arm. Values already set in the process environment take precedence.

The cross-encoder arm defaults to `Qwen/Qwen3-Reranker-8B` and uses `SILICONFLOW_RERANK_URL`, `SILICONFLOW_RERANK_MODEL`, and `SILICONFLOW_RERANK_INSTRUCTION` when set. Its provider cost fields can be configured with `SILICONFLOW_RERANK_COST_USD_PER_1K_TOKENS` or `SILICONFLOW_RERANK_RESERVED_USD_PER_REQUEST`.

```dotenv
TYPESAFE_API_KEY=your-key
```

For the BGE-M3 experiment, pass `--dense-backend local --navigation-model models/bge-m3` to every arm. This local model is used for domain navigation, file-title similarity, and Dense article retrieval. The local checkpoints are stored under `models/bge-m3` (BAAI revision `5617a9f61b028005a4858fdac845db406aefb181`) and `models/bge-small-zh-v1.5` (BAAI revision `7999e1d3359715c523056ef9478215996d62a620`). Use `--navigation-model models/bge-small-zh-v1.5` to select the latter explicitly. The CLI default remains `BAAI/bge-small-zh-v1.5`. To reproduce the remote BGE-M3 variant, use `--dense-backend siliconflow` and set `SILICONFLOW_API_KEY`.

The first run embeds all 55,348 corpus articles locally and reports progress. This is a one-time local computation; its vectors are cached for later arms and runs.

Corpus embeddings are stored locally under `.cache/legal-stage-class/` after the first successful build. The cache is ignored by Git and is never shared through the repository. A later `run` or `acceptance` command reuses it only when the dataset hash, full corpus text, SiliconFlow endpoint, model, encoding format, and input template all match. The arms in one `acceptance` process also share the same in-memory vectors.

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

For the BGE-M3 rerun, pass the same verified final question list to each arm:

```bash
legal-stage-class run \
  --dataset data/processed/stard/<dataset_hash> \
  --strategy baseline_hybrid \
  --query-set configs/stard_final_308_v6.json \
  --dense-backend local --navigation-model models/bge-m3
```

For PRE and POST, set `TYPESAFE_JEV_MODEL=jev-1.14.0` in `.env` to reproduce the reported model version.

The historical 600-query development set overlaps this final list on 112 queries. For live acceptance, use three IDs outside the new final list. This runs baseline, then PRE, then POST:

```bash
legal-stage-class acceptance \
  --dataset data/processed/stard/<dataset_hash> \
  --query-ids 1023,591,683 \
  --dense-backend local --navigation-model models/bge-m3
```

Add `--include-cross-encoder` to include the independent SiliconFlow `Qwen/Qwen3-Reranker-8B` arm:

```bash
legal-stage-class acceptance \
  --dataset data/processed/stard/<dataset_hash> \
  --query-ids 1023,591,683 \
  --dense-backend local --navigation-model models/bge-m3 \
  --include-cross-encoder
```

Each arm writes `manifest.json`, `rankings.jsonl`, `metrics.json`, `latency_cost.jsonl`, and `provider_ledger.jsonl` to `artifacts/runs/<run_id>/`. Jev requests from the project root share the persistent `artifacts/jev-budget.sqlite3` budget across arms, acceptance runs, and restarts. The $24 cap assumes the official TypeSafe endpoint, its standard context limit, and the $0.042 per million input token rate. Before each send, including transport retries, the CLI reserves the standard request ceiling. It reconciles against response input tokens and logs the per-call and cumulative estimates. Requests without usage retain their reservation. Cache hits make no paid call. Keep the database when resuming a run. The database tracks this project's Jev usage, and the account balance remains with TypeSafe.

## Implementation contract

The following behavior is shared across collaborators and should not change silently:

| Arm | Fixed behavior | Arm-specific behavior |
| --- | --- | --- |
| `baseline_hybrid` | Shared directory, complete BM25 and Dense rankings, RRF (`k=60`) | Returns the RRF top ten |
| `jev_pre_route` | Shared directory, metrics, and corpus vectors | Jev Choice selects two files. Both retrievers contribute complete rankings of the selected files' articles to RRF |
| `jev_post_rerank` | Same directory and RRF ranking as the baseline | Jev Choice reranks the top 30 article candidates and returns the top ten |
| `cross_encoder_rerank` | Same directory and RRF ranking as the baseline | SiliconFlow `Qwen/Qwen3-Reranker-8B` reranks the top 30 article candidates and returns the top ten |

PRE, POST, and cross-encoder reranking run as independent experimental arms. Use the same Dense backend for all arms in a comparison.

## Run-artifact contract

| File | What it contains | Development use |
| --- | --- | --- |
| `manifest.json` | Dataset hash, query IDs, parameters, provider-cost summary, cache source/key | Confirm two runs used comparable inputs |
| `rankings.jsonl` | One row per query: shortlist, final Top-10, ranks, PRE route or POST decision metadata | Recompute or inspect ranking changes |
| `metrics.json` | Macro retrieval metrics and per-query values | Compare arms only on the same query set |
| `latency_cost.jsonl` | Per-query end-to-end, domain routing, Hybrid retrieval, applicable PRE or POST stage times, and reserved/estimated cost | Separate online latency from model warm-up and index construction in `metrics.json` |
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

