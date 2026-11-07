# Experimental pipeline

The three proposed methods are parallel interventions at different points of a shared legal retrieval pipeline. They should be compared independently rather than applied sequentially.

```mermaid
flowchart TB
    Q[User consultation query]
    C[STARD statutory article corpus]

    subgraph PRE[Before retrieval: multi-law routing]
        P1[Jev relevance judgments over law descriptions]
        P2[Selected laws and article sub-corpus]
        P3[BM25 + dense retrieval]
        P4[RRF]
        P5[Top-k articles]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph BASE[Shared hybrid baseline]
        H1[BM25 candidates]
        H2[Dense candidates]
        H3[RRF]
        H4[Hybrid top-n shortlist]
        H1 --> H3
        H2 --> H3
        H3 --> H4
    end

    subgraph TAG[During indexing and retrieval: shared-concept tagging]
        T1[GPT concept vocabulary and multi-label annotations]
        T2[Fine-tuned Chinese BERT tagger]
        T3[Query/article concept tags]
        T4[Tag-overlap candidates]
        T5[BM25 + dense candidates]
        T6[Weighted RRF]
        T7[Top-k articles]
        T1 --> T2 --> T3 --> T4 --> T6 --> T7
        T5 --> T6
    end

    subgraph POST[After retrieval: Jev shortlist selection]
        J1[Query + hybrid top-n]
        J2[Jev n-way choice]
        J3[Selected article]
        J4[Ranking from option probabilities]
        J5[Top-k articles]
        J1 --> J2
        J2 --> J3
        J2 --> J4 --> J5
    end

    Q --> P1
    C --> P2
    Q --> H1
    Q --> H2
    C --> H1
    C --> H2
    C --> T1
    Q --> T3
    C --> T5
    H4 --> J1
    Q --> J1

    P5 --> E[Common evaluation]
    H4 --> E
    T7 --> E
    J3 --> E
    J5 --> E
```

## Experimental ceilings

- **Routing retention** measures whether relevant articles survive the pre-retrieval routing stage.
- **Shortlist Recall@n** is the upper bound for both Jev shortlist selection and a conventional cross-encoder reranker.
- Concept-tag quality affects both candidate coverage and ranking because tag overlap participates in candidate generation and weighted fusion.

## Recommended implementation order

1. Reproduce BM25, dense, and RRF baselines.
2. Freeze shared data splits and evaluation code.
3. Add each intervention as an independent experiment.
4. Tune prompts, thresholds, routing breadth, fusion weights, and shortlist size only on validation data.
5. Run final evaluation once on the reserved development split.

