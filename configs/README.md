# Configurations

[`stard_final_308_v6.json`](stard_final_308_v6.json) is the ordered question list extracted from `final_test_questions.md` (SHA-256 `744168de43ea220e44c85e81782ca130302db3ec75d0b087650173978538f524`). The 308 IDs and 512 question–article pairs were verified against the prepared STARD dataset `d2bc0bf33155184c150c400cf1b29e03230e0dfc5c0a098dac5fdb3a59020482`. Select it with `run --query-set configs/stard_final_308_v6.json`.

Experiment configurations will live here. Each run should record:

- data split version;
- retriever and model versions;
- prompts and classification thresholds;
- routing breadth;
- reciprocal-rank-fusion weights;
- shortlist size;
- random seed;
- latency and cost settings.

