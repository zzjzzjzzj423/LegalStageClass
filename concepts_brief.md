# Jev Interventions for Legal Retrieval

* **Author:** zijian
* **Date:** 2026-11-07

## Overview

`jev-interventions` 是 LegalStageClass 的阶段性交付，不是整个项目研究范围的替代。它交付一个共享混合检索基线，以及两个彼此独立的 Jev 实验臂：检索前法律路由和检索后候选重排。两个 Jev 干预不得叠加，所有差异都应能归因到单一干预位置。

共享基线读取 STARD，运行本地 BM25 和 SiliconFlow `BAAI/bge-m3` Dense 检索，通过 RRF 融合并输出 Top-k。检索前实验先用 TypeSafe Jev Noul 对法律进行独立相关性判断，再把允许的法条集合交给同一检索器。检索后实验读取共享基线的 Top-n shortlist，用 TypeSafe Jev Choice 产生确定性全序，再输出 Top-k。三条实验都计算通用检索指标，并保留可审计的排名、指标、逐阶段延迟/费用和 run manifest。

本 feature 不实现项目中的检索中概念标签链路，也不实现 cross-encoder 控制项。二者仍属于项目路线，只是延期到后续设计。官方 STARD dev 被定义为 reserved final evaluation；validation 只能从官方 train 派生，任何 prompt、阈值、路由宽度、候选深度、RRF 参数或 shortlist 大小都不得根据 dev 调整。

## Analysis

### Reviewer Status

| Reviewer lens | Status |
| --- | --- |
| `design_critic` | completed |
| `codebase_details` | completed |
| `zoom_out` | completed |
| `comprehension` | completed |
| `implementation_readiness` | completed |
| `structural` | completed |
| `inventive_catalyst` | completed |

七个 reviewer 均完成了 persona、项目材料和 artifact contract 的完整阅读，无 reviewer 失败或重试。

### 核心结论

当前概念的研究边界已经清楚，但还不能直接进入实现。仓库目前只有包骨架，没有 CLI、配置、数据加载、provider adapter、检索、指标、缓存或 artifact schema。正式设计必须先关闭以下 Blocker，不能把决定留给实现阶段。

#### Blocker 1：项目拓扑与 feature 拓扑需要明确分层

`README.md` 和 `docs/pipeline.md` 的完整项目包含检索前、检索中、检索后三种分类干预，以及 cross-encoder 控制项。本 feature 只包含一个共享基线和两个 Jev 干预。它们不是同一个“三条实验”集合。

本 brief 将 `jev-interventions` 定位为阶段性 slice：稳定 experiment ID 为 `baseline_hybrid`、`jev_pre_route`、`jev_post_rerank`；`during_tagging` 和 cross-encoder 保留为尚未实现的项目工作。配置必须使用互斥的 strategy，启动前拒绝同时启用 pre-route 和 post-rerank。

#### Blocker 2：STARD 没有现成的法律路由目录

现有 corpus 记录只有 article `id/name/content`，没有权威 `law_id`、法律描述或 article-to-law mapping。Reviewer 的初步字符串分析得到约 1,258 个法律名称，但这不是可作为实验输入的权威目录。司法解释、决定、规定、别名和历史版本会让运行时字符串切割产生静默错误。

正式设计必须先定义一份不可变、版本化、可审计的 law catalog，至少包含稳定 `law_id`、规范名称、提供给 Jev 的文本、article IDs、来源、生成规则版本、源数据哈希和异常清单。每篇 corpus article 必须恰好映射到一个路由单元，或触发显式失败；所有 qrel article 必须可映射。若 Jev 使用法律描述而非标题，还必须冻结描述来源、版本和许可。

#### Blocker 3：Jev 能力尚未证明支持当前实验

当前概念假设 Noul 能提供可用于 validation 阈值选择的连续且可比较信号，也假设 Choice 会返回所有选项的可用概率。这两个假设必须通过小规模、真实、限额授权的 provider contract probe 验证。

Probe 必须固定实际 SDK/API 版本、模型标识、请求和响应 schema、最大选项数、上下文上限、批处理语义、概率和 confidence 的含义、token usage、request ID、限流、超时和无效响应。还必须确认 `Noul` 是正式 TypeSafe 原语名称。

若 Noul 只有布尔结果，就不能声称使用概率阈值；主路由策略应改为可验证的布尔选择和固定宽度规则。若 Choice 只返回 winner 或 winner confidence，主重排策略应是 winner-first、其余保持 baseline 相对顺序，不能声称按完整概率重排。

#### Blocker 4：检索前实验必须隔离路由与检索器变化

在路由后的子语料上重建 BM25 会改变 IDF 和平均文长；先在全库取有限 Top-M 再过滤又不是真正的检索前路由。这两种做法都会破坏与共享基线的公平比较。

推荐的矛盾消解是分离“检索器定义”和“候选可见性”：全局只构建一次 BM25 统计和 Dense 向量；pre-route 只生成 `allowed_law_ids/allowed_article_ids` mask；BM25 保持全局 IDF，Dense 保持相同向量、相似度和候选深度；RRF 参数不变。55,348 篇法条的首版 Dense 可以优先采用可验证的 masked exact similarity，避免用固定 oversampling 后过滤来冒充硬过滤。

正式设计仍需明确这一推荐是否获批。若选择 per-law index，则必须把它命名为不同的 retrieval treatment 或敏感性分析，不能称为“同一检索器上的纯路由干预”。

#### Blocker 5：预路由调用量可能不可接受

逐查询逐法律判断的规模是 `queries × laws`。STARD 共 1,543 queries，官方 train/dev 初步核对为 1,235/308；若 law catalog 接近初步观察到的数量，判断次数会达到百万级。必须区分“独立 judgment”与“API request”：一次结构化批量请求可以包含多个独立 judgment，但只有 provider contract 证明批处理保持预期语义时才可使用。

任何 live 命令都必须先输出最坏请求数、judgment 数、input/output token、金额和墙钟时间估算，并要求显式付费授权。运行时必须有 `max_calls`、`max_tokens`、`max_cost` 和有限重试的硬熔断；达到上限时停止新请求，保存可恢复的 partial run，不能默认为零成本或继续超支。

#### Blocker 6：split 防泄漏需要机器强制

项目必须保存不可变 split manifest，包含上游数据版本或 commit、输入文件 SHA-256、train/validation/final query ID、生成算法、seed、生成器版本和交集检查。项目级名称使用 `final`，同时记录其上游名称 `dev`，避免把 upstream dev 当作普通开发集。

调参入口只能消费 train/validation，并对 final fail closed。Final-eval 命令只能消费已经锁定的配置 hash，不得包含 sweep 或选择循环。最终设计还要明确 final 的运行授权、允许的故障重跑条件，以及如何避免反复查看 final 结果后回头改 prompt 或参数。

#### Blocker 7：真实 live E2E 是独立完成门

离线 schema test、录制响应 contract test 和本地 fixture E2E 可以帮助开发，但都不能替代最终真实 provider 边界。最终任务必须从公共 CLI 进入，经过真实配置、STARD loader、索引、BM25、SiliconFlow Dense、所选独立 Jev 干预、RRF 或重排、指标和 artifact writer。

该任务必须写明允许发送的数据、代表性 query/article 范围、最大 calls/tokens/金额、并发、超时、重试、退出状态和证据路径。未获得用户明确授权或未完成 bounded live run 时，live E2E 状态只能是 incomplete，不能宣称完整 real-chain coverage。

### 重要但非阻塞的设计要求

- 共享 baseline 必须冻结分词器、BM25 `k1/b`、文本字段、Dense 输入模板、向量维度、归一化、相似度、各路 candidate depth、RRF 常数、缺失排名处理和稳定 tie-break。post-rerank 应引用确切 baseline shortlist artifact hash，而不是重新计算一个近似 baseline。
- Dense cache 必须内容寻址。Identity 至少包含 corpus/article 文本哈希、article 顺序、provider、模型和 revision、输入模板、预处理版本、维度与归一化。写入要原子化，支持完整性检查和中断恢复；同一配置的暖运行不得再次生成 corpus embedding。
- Jev rerank 必须是从固定 shortlist 和固定 provider response 到完整全序的纯确定性变换。任何 choice、probability、confidence 的组合公式、并列规则、缺失值和 fallback 都要预注册。
- 路由为空、provider 部分失败、超时、概率缺失和解析失败不得静默混入正常 treatment。默认 fail closed；如研究需要 fallback，必须成为显式策略，并单独报告 fallback rate 和结果。
- Recall@10、nDCG@10、Hit@1、Routing Retention 和 Shortlist Recall@n 必须给出逐 query 公式、macro/micro 聚合、多个 relevant articles、空 qrels、短列表和错误 query 的处理。Routing Retention 建议同时保留 any-retained、all-retained、fraction-retained，避免一个名称承载多种含义。
- 每个 run 使用不可变 run ID 和版本化 schema，状态至少区分 `running/partial/failed/complete`。rankings、metrics、latency/cost 和 manifest 必须共享 run ID，互相记录 checksum。metrics 应能由 rankings 和 qrels 独立重算。
- 逐阶段 ledger 要区分一次性 corpus embedding/indexing、query embedding、BM25/Dense、Jev、重试和 cache hit；报告 cold/warm latency、总 wall-clock、p50/p95/p99、实际或估算 token、币种与价格快照。README 中的 local peak memory 和 offline indexing cost 应纳入统一效率 artifact；暂不测量时使用 `not_measured`，不能写成 `0`。
- API keys 只从环境读取。manifest 和日志不得保存 `SILICONFLOW_API_KEY`、`TYPESAFE_API_KEY`、Authorization header 或异常对象中的凭证。原始 provider payload 是否持久化必须有明确的脱敏、路径和保留策略。
- 不需要在首版建设向量数据库、消息队列、分布式执行器或通用 provider 插件框架。一个共享 runner、两个小型 typed adapters、内容寻址的本地 artifact 和可恢复 checkpoint 足以覆盖当前范围。

### Ideal Final Result

冻结一次 corpus、law catalog 和 splits 后，用户通过同一 CLI 选择 `baseline_hybrid | jev_pre_route | jev_post_rerank`。三个 arm 共享同一数据、BM25、Dense、RRF、评价器和 artifact writer；PRE 只增加 law/article mask，POST 只对带 provenance 的 baseline shortlist 做确定性排序。昂贵计算以内容寻址的不可变 artifact 复用，远程调用在发出前即可证明成本有界，任何失败都可追溯和恢复，reserved final 在参数锁定后才可运行。

## Goals vs. Non-Goals

### In-Scope Goals

- 交付 `baseline_hybrid`、`jev_pre_route`、`jev_post_rerank` 三个可由真实 CLI 独立运行的 experiment arm。
- 建立共享、冻结、可复现的 BM25 + SiliconFlow `BAAI/bge-m3` Dense + RRF 基线。
- 首次生成后安全复用 corpus embeddings，并能证明缓存命中、失效和完整性。
- 建立版本化 STARD split manifest 和 law catalog；validation 仅从官方 train 派生，官方 dev 只用于 reserved final evaluation。
- 对 pre-route 记录法律判断、article allow-list、Routing Retention、检索质量、token、费用和延迟。
- 对 post-rerank 记录 baseline shortlist provenance、option 映射、choice/probabilities/confidence 原始字段、确定性全序、Shortlist Recall@n、检索质量、token、费用和延迟。
- 每次运行生成 versioned rankings、metrics、latency/cost 和 run manifest，完整记录数据/split、代码、模型/provider、检索器、prompt、阈值、seed、缓存和策略参数。
- 提供离线 focused checks、真实 parser/CLI 的代表性 fixture E2E，以及经明确授权和硬费用上限保护的 bounded live E2E。
- 保持两条 Jev 干预独立，配置和运行时都不能叠加。

### Out-of-Scope Non-Goals

- 本 feature 不实现 shared-concept tagging、GPT 概念词表、多标签标注或 BERT tagger；这些仍是项目后续独立干预。
- 本 feature 不实现 cross-encoder reranker；它仍是项目级后续控制项。
- 不把 pre-route 和 post-rerank 组合成一个 treatment。
- 不在官方 STARD dev 上选择 prompt、阈值、路由宽度、RRF 参数、candidate depth 或 shortlist size。
- 不把 lexical law prefilter、local embedding、per-law BM25、pairwise/tournament rerank 或失败时全库 fallback 静默加入主实验。
- 不建设通用法律知识图谱、向量数据库、服务化索引、分布式任务系统或任意 provider 插件平台。
- 小规模 smoke run 不代表完整 STARD 研究结果；离线 mock/录制响应不代表真实 live E2E 完成。

## Alternatives Weighed

### 把本 feature 当作整个项目的永久缩减

当前不采用。它会与 `README.md`、`docs/pipeline.md` 和仓库关于三种分类干预的约束冲突。本 feature 被定义为阶段性 slice；若以后决定永久删除 concept-tagging 或 cross-encoder，必须通过新的批准设计显式 supersede 项目来源。

### 每个 query 按路由结果临时重建 BM25

当前不采用为主实验。该方案会改变 IDF、平均文长和索引成本，使效果无法只归因于路由。它可以作为单独命名的敏感性分析，但不能冒充共享检索器上的 pre-route。

### 全库先取有限 Top-M，再按法律过滤

不采用。该方案不是检索前路由，并会因 Top-M 截断让允许法律中的相关法条永远无法进入候选。主方案需要真正的 doc allow-list 过滤或可由 exact oracle 验证的等价实现。

### 每部法律建立独立索引

暂不采用为主方案。它可能简化过滤和降低查询成本，但会改变 BM25 统计、合并语义和维护成本。只有在设计明确将其定义为不同 treatment，并给出跨法律合并规则时才可考虑。

### 先用 lexical/dense 方法筛选法律，再调用 Jev

暂不采用。它能显著降低 Jev 费用，但会引入新的候选召回上限和前置干预，改变当前“对每部法律独立判断”的研究问题。若成本证明主方案不可运行，应作为新的 experiment ID 重新设计。

### Noul 布尔输出上伪造概率阈值

不采用。只有真实 contract 证明返回连续且可比较的分数时才能调 threshold；否则使用布尔路由和明确的 min/max width 规则。

### Choice 只返回 winner 时推断完整概率排序

不采用。若 provider 不返回每个 option 的可靠同尺度概率，主方案使用 winner-first、baseline-stable-tail；confidence 只作观测或预注册的 fallback 门控。

### 分块、pairwise 或 tournament rerank

暂不采用。它们可绕过 n-way 上限，但会增加调用量，并引入跨块不可比和位置偏差。若单次 n-way 无法承载选定 shortlist，应作为新 POST 变体设计，而不是实现期补丁。

### provider 失败时静默回退 baseline 或全库检索

不采用为主实验。它会把 treatment 失败混入成功样本并污染质量、成本和延迟。默认行为是有限重试后生成 partial/failed run；任何 fallback 都必须显式配置并单独报告。

### 本地 embedding 替代 SiliconFlow 完成最终验收

不采用。它可用于离线开发和接口测试，但不能满足指定 provider 的最终实验条件，也不能替代经授权的 live E2E。

### 使用向量数据库或分布式执行框架

当前不采用。55,348 篇法条可用本地内容寻址向量文件、内存映射或精确相似度完成首版。只有测得真实瓶颈后再评估更复杂基础设施。

## Key Decisions & Staging for Next Steps

### 已确定的方向

1. `jev-interventions` 是阶段性交付：一个共享基线和两个独立 Jev 干预；不代表完整项目研究矩阵已完成。
2. 三个 experiment arm 使用互斥 strategy；pre-route 和 post-rerank 不得叠加。
3. 官方 dev 是 reserved final evaluation；validation 只从官方 train 派生，dev 不参与任何参数或 prompt 选择。
4. BM25 在本地运行；Dense 使用 SiliconFlow `BAAI/bge-m3`；Jev 使用 TypeSafe；密钥分别来自 `SILICONFLOW_API_KEY` 和 `TYPESAFE_API_KEY`。
5. corpus embeddings 必须缓存；每次 run 必须输出 rankings、metrics、latency/cost 和 manifest，并记录完整 provenance。
6. 最终完成门是经 CLI 的真实仓库链路与有界 live provider 调用；未经授权不能用 mock 代替。

### 进入正式设计前必须冻结的决策

1. **Law catalog：** 路由单元、规范名称、给 Jev 的文本、article mapping、来源、版本和人工审计方法。
2. **Provider contract：** Noul/Choice 的准确 API、模型版本、batch 和 n-way 上限、概率/confidence 含义、usage 与错误 schema。
3. **Pre-route 检索：** 是否批准“全局 BM25 统计 + article allow-list + 同一 Dense/RRF”的推荐语义，以及过滤实现的 exact oracle。
4. **Routing policy：** score 域、threshold 与 max width 顺序、min width、并列规则、zero-route 和 partial failure 行为。
5. **Post-rerank policy：** 完整概率可用时的排序公式；不可用时的 winner-first 规则；confidence 的唯一用途；option 顺序、文本截断和 tie-break。
6. **Baseline contract：** tokenizer、文本字段、BM25 参数、Dense 模板/维度/归一化、candidate depths、RRF 常数、final k、shortlist n 和稳定排序。
7. **Split policy：** validation 比例/分组方式、seed、物化 ID、hash、final lock 和合法重跑规则。
8. **Metric definitions：** 每项指标的公式、聚合、空集和多相关法条语义，以及 paired delta/置信区间是否纳入正式报告。
9. **Artifact/cache schemas：** 字段、版本、checksum、不可变 run 目录、原子提交、partial 状态、resume 和兼容/迁移策略。
10. **Cost and authorization：** dry-run 估算方式、数据外发授权、最大 query/article/calls/tokens/金额、并发、重试、熔断和证据保留。

### 建议的后续阶段顺序

1. 先做只读数据审计和小型 provider contract probe，产出 law catalog 方案、真实请求/响应能力表和预路由成本上界；probe 需要用户明确授权和小额硬上限。
2. 用 `egm-draft` 把本 brief 写成 `docs/designs/jev-interventions.md`，逐项关闭上述十个决策，并明确 acceptance scenarios 和未解决事项。
3. 设计获批后，用 `spec-to-plan` 生成 `docs/plans/` 下的兼容计划。计划应先实现并冻结 baseline 与 split，再实现 law catalog/pre-route，再实现 post-rerank，最后完成统一 artifact、恢复和 live acceptance。
4. 最终计划任务必须从真实 CLI 分别覆盖 `baseline_hybrid`、`jev_pre_route`、`jev_post_rerank`，使用代表性数据和确定性 seed，检查策略隔离、cache warm hit、排名/指标/账本/manifest schema、provenance、预算熔断和恢复。
5. 只有 focused checks 与经授权 live E2E 都成功、证据被人工检查后，才能报告该实现计划完成。若 live 授权仍缺失，保留 incomplete gate，不得降级完成标准。
