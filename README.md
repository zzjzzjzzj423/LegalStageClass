# LegalStageClass

**Where to Classify? Comparing Classification Before, During, and After Legal Retrieval**

LegalStageClass is a research project studying where classification is most useful in a Chinese legal retrieval pipeline. Using the STARD benchmark, the project compares three classification-based interventions against a shared hybrid retrieval baseline.

## Research question

How do classification-based interventions at different stages of hybrid legal retrieval affect retrieval quality and computational cost?

## Compared strategies

1. **Before retrieval - multi-law routing**  
   Jev selects one or more relevant laws before BM25 and dense retrieval search the corresponding article collections.

2. **During indexing and retrieval - shared-concept tagging**  
   A Chinese BERT classifier assigns shared legal concepts to queries and articles. Tag-overlap candidates are fused with BM25 and dense candidates using weighted reciprocal rank fusion.

3. **After retrieval - Jev shortlist selection**  
   Jev receives the hybrid top-n shortlist, chooses the most relevant article, and uses option probabilities to rerank the shortlist.

Each strategy is evaluated independently on top of the same hybrid baseline rather than being stacked into one system.

## Baselines and controls

- BM25-only retrieval
- Dense-only retrieval
- BM25 + dense retrieval with reciprocal rank fusion
- Hybrid retrieval followed by a pretrained Chinese cross-encoder reranker

## Evaluation

Retrieval quality:

- Recall@10
- nDCG@10
- Hit@1
- Routing retention
- Shortlist Recall@n

Efficiency:

- Per-query latency
- API tokens and fees
- Local peak memory
- Offline annotation, training, and indexing cost

## Repository layout

```text
configs/                 Experiment configuration notes
data/                    Dataset setup documentation (data is not committed)
docs/pipeline.md         Research pipeline and experimental design
proposal/                Original project proposal
src/legal_stage_class/   Implementation package
tests/                   Tests
```

## Project status

Proposal and experimental design stage. Implementation will follow the shared-baseline-first order described in [docs/pipeline.md](docs/pipeline.md).

## Data

The project uses STARD: *A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals* (Findings of EMNLP 2024).

- Paper: <https://aclanthology.org/2024.findings-emnlp.625/>

