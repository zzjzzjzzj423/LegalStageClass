# Data

This project uses the STARD Chinese statute retrieval dataset.

Dataset files are intentionally excluded from version control. Place downloaded source data under `data/raw/` and derived artifacts under `data/processed/`.

The final evaluation split must remain untouched while prompts, thresholds, and fusion weights are selected on the validation split.


## Corpus name audit (2026-11-08)

Downloaded the public `data/corpus.jsonl` from the upstream STARD repository and applied the exact `ARTICLE_SUFFIX` parsing and name grouping used by `data/processed/prepare_stard.py`.

- Corpus records: **55,348**; distinct extracted document names: **1,258**; names failing the parser: **0**.
- This corrects the earlier **1,257** grouping count recorded in `concepts_brief.md`. These are document-name groups, not the number of current national laws.
- Upstream describes its corpus as national-level laws, regulations, and judicial interpretations: https://github.com/oneal2000/STARD#data-collection-methodology

The following mutually exclusive buckets are based only on names. First collect names beginning with `最高人民法院` or `最高人民检察院`; then classify remaining names by their ending. This is a descriptive audit, **not a classification of legal authority or current validity**.

| Name rule | Document groups | Article records |
| --- | ---: | ---: |
| Starts with 最高人民法院 / 最高人民检察院 | 335 | 7,450 |
| Remaining names ending in 法 | 274 | 19,593 |
| Remaining names ending in 条例 | 438 | 18,818 |
| Remaining names ending in 办法 | 99 | 2,874 |
| Remaining names ending in 规定 | 81 | 2,492 |
| Other names | 31 | 4,121 |
| Total | 1,258 | 55,348 |

For example, `劳动合同法` (98 articles), `食品安全法实施条例` (86), `道路交通安全违法行为记分管理办法` (37), and `最高人民法院关于审理票据纠纷案件若干问题的规定` (75) are separate groups. The other-name bucket includes `中华人民共和国民法典` (1,260 articles), so the 274 names ending in 法 must not be presented as the exact number of laws in the corpus. Conversely, a name ending in 条例 or 办法 does not alone establish its legal authority.

This dataset snapshot is not a current-validity directory: the audit does not validate enactment dates, repeals, aliases, or versions. Comparing its total to a dated official current-law count is not a like-for-like comparison.

Corpus SHA-256: `15c77a1333d947ec88cf4cb4c99d70dafd946ec91d35f2bbc77824817c7f7fa4`.

## Gold-label audit (2026-11-08)

This is a descriptive audit of the full downloaded `queries.json`, not a rule for selecting candidates or tuning experimental parameters. Join each `match_id` to `corpus.jsonl`, deduplicate IDs per query, and extract document names using the existing article-suffix rule. All referenced IDs exist in the corpus; no query has duplicate gold IDs. `match_name` agrees with the joined names as a set for every query, although ordering differs for 221 queries. Gold list order must not be treated as a relevance ranking.

| Statistic | Result |
| --- | ---: |
| Queries | 1,543 |
| Query–gold-article pairs | 2,717 |
| Mean gold articles per query | 1.761 |
| Queries with exactly one gold article | 843 (54.6%) |
| Queries with multiple gold articles | 700 (45.4%) |
| Maximum gold articles per query | 23 |
| Queries with gold articles from one document-name group | 1,249 (80.9%) |
| Queries spanning multiple document-name groups | 294 (19.1%) |
| Distinct document-name groups appearing in gold labels | 133 of 1,258 |
| Queries with at least one Civil Code article | 861 |
| Query–gold-article pairs from the Civil Code | 1,382 of 2,717 (50.9%) |

Gold article count distribution (count: queries): 1:843, 2:435, 3:157, 4:62, 5:26, 6:7, 7:9, 8:1, 9:1, 10:1, 23:1. Gold document-name count distribution: 1:1249, 2:258, 3:32, 4:4.

Using only the exact name-prefix test `最高人民法院` / `最高人民检察院` (not a verified legal-type classifier): 308 queries contain such gold articles; 153 have only such articles, 155 mix them with articles from other documents, and 1,235 contain none. Do not relabel these figures as exhaustive judicial-interpretation counts.

Examples from the dataset:

- Query 0, `谁可以成为个体工商户？`: Civil Code Article 54 plus Individual Businesses Regulation Article 2.
- Query 20, `土地承包经营方弃耕，村委会可以重新发包吗？`: Article 6 of the SPC interpretation on rural land contracting disputes only.
- Query 87, `妻子未经丈夫同意，能否自行终止妊娠？`: Civil Code Article 1003 plus Article 23 of the SPC interpretation (I) on the marriage and family book of the Civil Code.

The STARD paper Sections 3.1–3.3 describe narrowing legal subject areas, translating everyday facts into legal facts, and selecting a small sufficient set of provisions. Section 4.4 requires agreement between two annotators. No blanket source-type priority is specified. Gold labels are relevance sets, not legal-authority rankings or an exhaustive declaration that every unlabelled article is legally inapplicable.

For comparable STARD retrieval, retain the full corpus as the candidate universe. Searching that universe does not require reading all articles through a remote model on every query: use prebuilt retrieval indexes. Query-based routing is an experimental alternative, with retained-gold coverage measured before retrieval. Do not restrict candidates to the 133 gold-bearing document groups identified using the full dataset: that would use evaluation labels to determine the search space.

Sources: https://aclanthology.org/2024.findings-emnlp.625.pdf ; https://github.com/oneal2000/STARD/blob/main/data/queries.json

Queries SHA-256: `e64dcdbda5d2b734a7242253620dcf5e8325a3d726f0ca0a86e9b3664b8ba644`.

## 8 大类 / 22 领域分类验收（2026-11-08）

按用户确认的方案，保留现有 1,258 个文件名称组，在其上增加内容领域多标签。此次交付是本地数据分类与离线审计产物，不改变现有三条实验策略，不调用 Jev。分类依据是语料名称规则和明确的名称修正，标准答案只参与验收，不参与制定文件标签。

两阶段代码、分类结果和说明文档统一放在 `data/processed/`。代码和结果使用英文文件名，说明文档使用中文文件名。

两阶段职责如下，1,258 是当前 STARD 快照的实际分组数，并非代码写死的常量：

| 阶段 | 代码 | 输入与职责 | 输出 |
| --- | --- | --- | --- |
| 第一阶段：按文件名称归组 | `data/processed/prepare_stard.py` | 规范化原始数据，去掉法条名称末尾条号，将当前 55,348 条条文归成 1,258 个文件名称组 | 标准化数据目录中的 `law_catalog.jsonl`，以及查询、语料、标注和划分文件 |
| 第二阶段：按内容领域分类 | `data/processed/classify_domains.py` | 调用第一阶段的 `group_articles_by_document`，再为文件组分配 8 大类、22 个领域 | `file_domains.jsonl` 和中文规则、验收报告 |

第二阶段不会复制第一阶段的前缀解析实现。Git 放行两个处理脚本、包声明和下面两个生成文件，其余派生数据及原始语料仍然忽略。

| 文件 | 内容 |
| --- | --- |
| `data/processed/file_domains.jsonl` | 全部 1,258 个文件的领域归属、所含法条 ID 和匹配依据 |
| `data/processed/数据集分类规则.md` | 中文分类规则和验收报告；各领域数量、每领域随机 5 条样例、覆盖检查、范围测算和数据哈希 |

在仓库根目录按需运行：

```bash
# 第一阶段：生成标准化数据和文件名称分组
legal-stage-class data prepare --raw-root data/raw/STARD

# 第二阶段：生成领域分类与验收报告（无需先运行上面的 CLI）
python data/processed/classify_domains.py
```

代码只生成上述 JSONL 和 Markdown。逐条展开的领域表、重复 CSV、规则副本、独立样例文件、指标 JSON、逐题临时明细及此次独立核对脚本已清理；分类规则、样例、汇总、数据哈希和必要案例保留在代码与报告中，可重新计算。名称前缀分组函数现位于 `data/processed/prepare_stard.py`，第二阶段直接导入调用。

### 本次结果与解释

- 所有 1,258 个文件、55,348 条条文均有归属；340 个文件属于多个领域。各领域数量允许重叠；大类数量对条文 ID 去重，不能直接把各行相加。
- 样例采用种子 `20261108`，每领域先随机选 5 份不同文件，再各抽一条；名称、正文、领域归属均已对照原始数据核验。
- 对 1,543 个问题、2,717 个“问题—标准条文”配对做覆盖检查。每题使用**标准答案全部条文所带的领域标签并集**作为已知领域，随后取这些领域的全部候选条文并集，不挑选更小的标签组合。
- 按该口径，每题平均剩余 **4,537.41** 条，平均少搜 **50,810.59** 条，即 **91.80%**；标准条目遗漏 **0**。这是已知正确条文归属后的范围测算，覆盖 100% 只说明分类结构没有丢失这些标准条目，**不代表自动选领域的准确率或实际检索召回率**。
- 另检查强制只选一个领域的限制：即使根据标准答案挑选最能保留答案的单领域，仍有 **129** 个问题会遗漏，共遗漏 **146** 个标准条目配对。因此后续领域选择应允许多选。
- 本次仅完成文件级规则分类和抽样校正，未做法律专家逐条语义复核。综合文件整份继承标签会带入其他主题条文；例如《民法典》整份进入民事、婚姻、住房领域，不能据此宣称婚姻领域的每条候选都与婚姻有关。
- 独立验证逐项核对文件与条文覆盖、22 个领域的数量、110 条样例来源、全部标准条目及逐题候选并集；另核验“人民法院”误命中“民法”、“人民政府”误命中“民政”、“文化遗产”误命中“继承遗产”等已发现的标题边界问题。

两个处理脚本现位于 `data/processed/`，由 `data.processed` 包提供导入；原有数据准备 CLI 和实验加载器已同步更新。包声明文件仅用于 Python 导入，没有复制处理实现。
