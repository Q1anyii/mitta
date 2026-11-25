# Mitta 评测报告与测试数据集分析

分析范围：RAGAS 五指标、记忆层、检索缓存层、MCP 工具层。
数据截至：2026-10-28（`src/agent_test/memory_eval_report.json` 为该日最新报告）；其余报告按归档日期标注。
整理原则：所有数字均回读原始 JSON，标注文件名、日期与样本量；同一指标存在多套口径时分别列出，不跨口径比较。

---

## 一、结论摘要

1. **RAG 召回是最突出的系统性短板。** 21 条项目题上 answer_correctness 最低的 4 条（0.20/0.20/0.25/0.40）同时伴随 context_recall 低（0.13/0.20/0.25/0.40）；faithfulness 反而均为 1.00。说明模型没有编造，是"没检索到 → 按引用约束拒答/少答"。
2. **记忆层的弱点是排序，不是召回。** top1 命中 40%，top3 80%，top5 100%；延迟已压到毫秒级（写 1.77 ms、读 1.34 ms）。类别语义混淆（"习惯"被判成"偏好"、"工具"排到第 4）是主要扣分点。
3. **缓存层同义命中强、规模扩展弱。** 12 组同义改写 35/36 命中（97.2%），hard/unrelated 负样本误命中 0；但库存 12 条、topk=3 时降到 61.1%，topk 提到 12 可回升到 88.9%；长列表 KNN 首名命中仅 36.1%，4/36 完全找不到。
4. **MCP 工具层机制扎实、筛选口径失真。** 装配 6/6、安全 11/11、截断 13/13 全过；工具筛选 recall 0.8939、zero_hit 0，但 precision 0.1406 已接近 top_k=12 的算术上限 0.1477（期望工具平均 1.77 个），"精度低"主因是候选数过大，次因是 `web_search`、`write_query` 等动作边界误选。
5. **口径风险必须写进评测报告/对外说明。** `ragas_judge_report.json` 是自写 DeepSeek judge，不是 RAGAS 官方库；judge 把上下文截断到 3000 字；09-19 记忆报告 p95 < avg 且 count 仅 5/20，口径可疑；RAGAS 21 条与 28 条是两套题集，不可混用。
6. **修复轨迹可复现。** Send 扇出题：21:11 probe 得 0.20（当时知识库确实无此节）→ 21:26 提交新增 §9 → 22:24 重建后仍 0.00 → 22:53 after_fix 升到 1.00。证明瓶颈在检索/索引链路，不是语料缺失。

---

## 二、数据与口径

### 2.1 测试数据集

| 数据集 | 路径 | 规模 | 构成 | 用途 |
|---|---|---:|---|---|
| `eval_dataset.json` | `resources/knowledge-base/test-qa/` | 45 条 | 刁钻 Badcase 15、基础概念 10、代码调试 10、架构设计 10；字段 question / ground_truth / category | 通用 RAG 与 28 条 probe 的题源 |
| `eval_project_dataset.json` | 同上 | 21 条 | 10 个 category：langgraph 4、rag 5、security 3、system 3，其余 6 类各 1；含 key_points | 09-19 项目 RAGAS 报告题集 |
| `_probe_question_pool.json` | 同上 | 33.7 KB | 试探题库 | 本次未直接引用 |
| 记忆事实池 | `src/agent_test/eval_memory.py` | 15 条事实 / 10 条 query | 职业、技能×2、项目、知识、偏好×2、环境、位置、目标、学习、兴趣、工具×2、习惯 | 记忆检索质量与延迟 |
| 缓存用例 | `eval_cache_hitrate.py` | 12 组 × 3 改写 = 36；负样本 8+6 | 技术类同义改写、混合库存 1/3/6/12 条、hard/unrelated negatives | 命中率、KNN、embedding 成本 |
| 工具筛选用例 | `eval_tool_filter.py` | 22 条 / 41 个工具 | 期望工具 1–3 个，top_k=12 | 筛选 recall/precision |

### 2.2 报告清单

| 日期 | 文件 | 脚本 | 样本量 | 关键结果 |
|---|---|---|---:|---|
| 09-19 | `reports/2026-09-19/ragas_judge_report.json` | `eval_ragas_judge.py` | 21 | cp .6381 / cr .8005 / faith .959 / rel .9881 / corr .7976 |
| 09-23 21:11 | `src/agent_test/ragas_judge_probe_28.json` | 同上 | 28 | .6304 / .6154 / .9411 / .9875 / .6518 |
| 09-23 22:24 | `src/agent_test/ragas_judge_probe_after_rebuild.json` | 同上 | 28 | .6214 / .6614 / .9586 / .9593 / .6639 |
| 09-23 22:53 | `src/agent_test/ragas_judge_probe_after_fix.json` | 同上 | 28 | .6593 / .7432 / .9464 / .9929 / .7575（cp、cr、rel、corr 为三次最高；faithfulness 低于 after_rebuild 的 .9586） |
| legacy | `reports/legacy/ragas_summary.json` | `ragas_eval.py`（官方库） | 3 | .6384 / .5000 / .9167 / .3906 / .6892；threshold .3；7.97 s |
| 09-20 | `reports/2026-09-20/memory_eval_report.production.json` | `eval_memory.py` | 写 80×3、读 300×3、warmup 10 | 写 1.768 ms / 读 1.341 ms；去重 0.5 |
| 10-28 | `memory_eval_report.json` | 同上 | 10 条检索 query | top1 .4 / top3 .8 / top5 1.0；过期通过；degraded=false（日期按文件 mtime） |
| 09-19 | `reports/2026-09-19/cache_hitrate_eval_report.json` | `eval_cache_hitrate.py` | 36 改写 + 102 混合 | 同义 .9722；混合 80/102；负样本 0 误命中 |
| 09-18 | `reports/2026-09-18/tool_filter_eval_report.json` | `eval_tool_filter.py` | 22 | recall .8939 / precision .1406 / zero_hit 0 |
| 09-19 | `tool_assembly_eval_report.json` | `eval_tool_assembly.py` | 6 | 6/6 |
| 09-19 | `tool_safety_eval_report.json` | `eval_tool_safety.py` | 11 | 11/11 |
| 09-19 | `tool_truncation_eval_report.json` | `eval_tool_truncation.py` | 13 | 13/13 |
| 09-18 / legacy | `semantic_cache_eval_report.json`、`cache_eval_report.json`、`cache_ttl_eval_report.json` | 对应脚本 | 3 / 20×3 / 10 | 同义 3/3；热缓存命中 1.0；TTL 0% vs 33.3% |

### 2.3 口径风险（引用前必读）

1. **两套 "RAGAS"。** 只有 `reports/legacy/ragas_summary.json` 调用官方 `ragas.evaluate`，样本仅 3 条，且 `distance_threshold=0.3`，只能作历史参照。现行报告 `ragas_judge_*` 是自写 DeepSeek LLM-as-judge（temp=0），指标同名但实现不同，`reports/README.md` 已明文标注"**不是** RAGAS 官方库"。
2. **judge 输入被截断。** `eval_ragas_judge.py` 的评分 prompt 取 `context[:3000]`、`answer[:2000]`。当相关片段排在上下文 3000 字之后时，context_recall 会被低估；因此低 recall 要结合回答中"知识库暂未覆盖"的表述判断，不能只看分数。
3. **题集不同。** 09-19 是 21 条项目题；三个 probe 是 28 条通用题（来自 `eval_dataset.json`）。两者题面、category 分布均不同，跨文件比较会得到"同集不同分"的质疑。
4. **09-19 记忆报告口径可疑。** `reports/2026-09-19/memory_eval_report.json` 写延迟 avg 7.12 ms 但 p95 4.56 ms、读 avg 2.59 ms 但 p95 2.49 ms；延迟分布通常右偏，p95 低于均值不合常理，且 `count` 只有 5/20，样本口径不完整。该份数据不再引用，统一改用 09-20 生产口径。
5. **工具 precision 有算术上限。** 22 条用例期望工具平均 1.77 个，top_k=12；理论上限为 `Σ min(e_i,12)/12/22 = 0.1477`。实测 0.1406 已达上限的 95.2%，不能读成"86% 的候选是噪声所以筛选很差"。
6. **小样本。** 10（记忆检索）、21/28（RAGAS）、22（工具筛选）、6/11/13（工具机制）均未做置信区间；结论只能描述这批题，不宜外推为全量业务指标。

### 2.4 其他报告中的异常（不在四层主表内）

- **限流实际未拦截（legacy）。** `reports/legacy/rate_limit_eval_report.json`：`over_limit` 40 条请求、阈值 30、应拦 10 条，实际 `blocked=0`、`intercept_accuracy=0.75`；而 `normal_traffic` 30/30 通过、`path_filtering` 1.0、`concurrent_stress` 100000 并发拦截 99931 条（99.9%）。同一份报告内"超限不拦"与"并发全拦"互相矛盾，若报告要写限流，不能用"通过"概括，需先查 `over_limit` 用例的判定逻辑。
- **09-23 的检索参数试验未纳入。** `probe_no_rewrite`、`probe_no_rerank_sort`、`retrieval_eval_rrf_weights*` 均只输出检索指标，文件 2 KB 量级，没有与 21/28 题集可比的五指标；本报告未采用其数值。

---

## 三、RAGAS 五指标

### 3.1 总览

09-19 报告配置：`limit=21`、`n_results=20`、`filter_threshold=0.15`、`judge=deepseek-chat`（`ragas_judge_report.json` 的 config 字段）。

两列来自不同题集，仅并列展示各自分数，不比较差值。

| 指标 | 09-19 项目集 n=21 | 09-23 after_fix n=28 | 含义 |
|---|---:|---:|---|
| context_precision | 0.6381 | 0.6593 | 上下文中相关片段占比，噪声越少越高 |
| context_recall | 0.8005 | 0.7432 | 标准答案事实点被上下文覆盖的比例 |
| faithfulness | 0.9590 | 0.9464 | 回答事实被上下文支撑的比例，抗幻觉 |
| answer_relevancy | 0.9881 | 0.9929 | 答问语义相关度 |
| answer_correctness | 0.7976 | 0.7575 | 与标准答案的事实匹配度 |

按 category 看（09-19，n=21）：`rag-retrieval-system` cp 0.44、`langgraph-architecture` cp 0.588、`frontend-vue3` cp 0.40 是精度洼地；`security-authentication` corr 0.717、`system-architecture` corr 0.75 是正确性洼地。`exception-handling`、`engineering-practices` 五项均 ≥0.8。

### 3.2 逐指标 bad case / 高指标 case

| 指标 | bad case（分数） | 高指标 case（分数） |
|---|---|---|
| context_precision | `项目的 LangGraph 检索图是怎么构建的？节点有哪些？` 0.30；`项目知识库文档入库时怎么切分的？` 0.25；`LangGraph 的 Send 扇出机制…` after_rebuild 0.05 | `配置管理怎么做的？敏感信息怎么脱敏？` 1.00；`密码是怎么存储的？` 1.00；`AI Agent 项目的目录结构…` probe 三次均 1.00 |
| context_recall | `项目知识库文档入库时怎么切分的？` 0.13；`项目的 LangGraph 检索图…` 0.20；`密码是怎么存储的？` 0.25；`配置管理…` 0.40 | `前端 401 时怎么处理？` 1.00；`全局异常处理器怎么设计的？` 1.00；`JWT 双 Token 机制是怎样的？` 1.00 |
| faithfulness | `SSE 流式响应是怎么实现的？` 0.75（21 条最低）；`项目怎么集成 MCP 服务器的？` 0.80；`为什么还要做会话归属校验？` 0.80；after_rebuild 中 `FastAPI 中间件如何实现请求限流？` 0.65、`JWT + Redis 双 Token…` 0.78 | 21 条中 15 条为 1.00、18 条 ≥0.9，含切分、节点、密码三条低分题与 401 |
| answer_relevancy | `项目知识库文档入库时怎么切分的？` 21 集 0.80、probe_28 0.70；`LangGraph 的 Send 扇出…` probe_28/after_rebuild 0.80；`如何设计自定义异常类…` after_rebuild 0.90 | 21 条中 19 条为 1.00、20 条 ≥0.9 |
| answer_correctness | `项目的 LangGraph 检索图…` 0.20；`项目知识库文档入库时怎么切分的？` 0.20；`密码是怎么存储的？` 0.25；`配置管理…` 0.40 | `前端 401 时怎么处理？` 1.00；`SSE 流式响应…` 1.00；`全局异常处理器…` 1.00；`MCP 服务器集成` 1.00；`JWT 双 Token` 1.00；`距离阈值过滤` 1.00 |

### 3.3 重点 bad case 拆解

**（1）文档切分（cp 0.25 / cr 0.13 / faith 1.00 / rel 0.80 / corr 0.20）**

标准答案四个事实点：`MarkdownHeaderTextSplitter`（按 #、##）、`RecursiveCharacterTextSplitter`、`chunk_size=500`、`chunk_overlap=50`。
模型回答只说出"`.md` 按标题切分、`.txt/.pdf` 原样加载"，随后称"'### 2.3 文档切分策略'只有标题、没有正文内容，知识库暂未覆盖"。
但源文件 `resources/knowledge-base/04-rag-retrieval-system.md` 第 69–88 行确有完整代码与三条切分原则。**结论：知识库有答案，进入上下文的 chunk 只有标题、正文未进 top-20；最可能是标题与正文被切到不同块，属切片/召回问题。faithfulness 1.00 也证明模型没有编造。**

**（2）LangGraph 检索图节点（cp 0.30 / cr 0.20 / faith 1.00 / rel 1.00 / corr 0.20）**

标准答案：`StateGraph` + `RAGState`，六个节点 check_cache / rewrite / retrieve / rerank / store_cache / output_node。
模型回答："资料不足以列出该检索图的具体节点清单，尤其没有给出 `add_node`、`add_edge`…"。
源文件 `03-langgraph-architecture.md` 第 113–119 行逐行列出六个 `add_node`，且该内容自 2026-08-21 已存在（git 74efd3d）。**结论：不是语料更新问题，是检索未命中代码块。**

**（3）密码存储（cp 1.00 / cr 0.25 / faith 1.00 / rel 1.00 / corr 0.25）**

模型答出 `passlib.context.CryptContext` + `bcrypt` + `pwd_context.hash()`，并主动说明"哈希存哪个字段资料未覆盖"。
标准答案还要求：自动加盐、`rounds=12`、新密码至少 6 位、改密后清 Redis token。**precision 满分而 recall 只有 0.25，是"检索到的块很准但只覆盖一个片段"的典型：同一主题的多段事实没有被一起召回。**

**（4）SSE 流式响应（faith 0.75，为 21 条中该指标最低；cp 0.85 / cr 0.75 / rel 1.00 / corr 1.00）**

回答开头写"SSE 是服务器向客户端单向推送事件的 HTTP 协议，基于 HTTP 长连接"，这句属于文档外通识；后半段 `StreamingResponse`、`text/event-stream`、`yield data:` 均有 [文档] 角标。**答案整体正确（corr 1.00），扣分仅因掺入未标注的常识句。**这是"答案对但 faithfulness 被通识内容拉低"的边界 case。

**（5）前端 401（cp 0.40 / cr 1.00 / faith 1.00 / rel 1.00 / corr 1.00）**

答案完整覆盖 localStorage、Bearer、401 跳转、清缓存；precision 只有 0.40 是因为 top-20 检索结果中相关片段占比低，混入了无关片段。**同一 case 同时是 precision 的 bad case 与其余四项的高指标 case，说明精度低不一定影响答对。**

### 3.4 重点高指标 case

- `全局异常处理器怎么设计的？`：cp 0.8 / cr 1.0 / faith 1.0 / rel 1.0 / corr 1.0。答案结构为"最后一道防线—@exception_handler(Exception)—logger.exception—JSONResponse 500"，每个事实点都能在上下文找到角标。
- `AI Agent 项目的目录结构应该如何设计？`（after_rebuild / after_fix）：cp 1.0 / cr 0.89 / faith 1.0 / rel 1.0 / corr 0.89（probe_28 时 cr 与 corr 仅 0.33）。单点、可直接检索的目录事实，chunk 粒度与问题粒度匹配。
- 共性：高指标题集中在"单点配置、目录、异常处理、认证流程"这类**事实密度低、段落完整**的问题；低分题集中在"代码块清单、参数细节、阈值/轮数"这类**事实密度高、跨段分布**的问题。

### 3.5 修复轨迹（28 条口径）

| 时间 | 文件 | Send 扇出 cp/cr/corr | 总体 cr / corr |
|---|---|---|---|
| 09-23 21:11 | `ragas_judge_probe_28.json` | 0.10 / 0.15 / 0.20 | 0.6154 / 0.6518 |
| 09-23 21:26 | git 263fd4c 新增 `03` §9 Send（58 行） | — | — |
| 09-23 22:24 | `..._after_rebuild.json` | 0.05 / 0.00 / 0.00 | 0.6614 / 0.6639 |
| 09-23 22:53 | `..._after_fix.json` | 0.60 / 1.00 / 1.00 | 0.7432 / 0.7575 |

重建后 Send 反而从 0.20 掉到 0.00，而新增 §9 已随 263fd4c 写入源文件；after_fix 又恢复正常。可判定：**22:24 那次重建没有把新增 §9 有效纳入召回（旧 chunk 清理、索引切换或阈值过滤存在不一致），不是知识库缺内容。**具体是 ingest 增量逻辑还是 rerank 阈值所致，需查那次重建日志，本报告不替它下结论。

---

## 四、记忆层

### 4.1 指标总览

| 指标 | 口径 | 结果 |
|---|---|---|
| 写入延迟 | 09-20 生产，80 次×3 轮，warmup 10 | 汇总 avg 1.768 ms / p50 1.571 / p95 3.013 / max 3.745；单轮最高 6.024（第 2 轮） |
| 检索延迟 | 09-20 生产，300 次×3 轮 | 汇总 avg 1.341 ms / p50 1.172 / p95 2.086 / max 7.581；单轮最高 50.56（第 1 轮） |
| 检索质量 top1 | 10-28，10 条 query，threshold .5，top_k=5 | 0.40 |
| 检索质量 top3 | 同上 | 0.80 |
| 检索质量 top5 | 同上 | 1.00 |
| 去重写入 | 09-20，3 轮 5 写 5 跳 | reduction_rate 0.50 |
| 会话画像增长 | 09-20，10 轮 | entry 1→10 线性；写 avg 2.176 / p95 4.778；读 avg 1.152 |
| 过期清理 | 10-28 | expired_hidden=true，keep_visible=true |
| 降级标记 | 10-28 | degraded=false（Redis 后端正常） |

注：报告顶层汇总字段并非同一轮统计量。写延迟的 avg 取第 1 轮、p50 取第 2 轮、p95 取第 3 轮、max 取第 1 轮；读延迟的 avg/p50/p95/max 均取第 3 轮。引用单轮结论时以 `runs[]` 为准，不要把这些汇总值当成同一批 240/900 次采样的分布。

### 4.2 逐指标 bad case / 高指标 case

| 指标 | bad case | 高指标 case |
|---|---|---|
| top1 命中 | `用户熟悉哪些技术栈？` 首位"学习"，"技能"第 3；`用户做过什么项目？` 首位"位置"，"项目"第 3；`用户的学习方向？` 首位"职业"；`用户的职业背景？` 首位"位置"；`用户用什么工具部署？` 首位"目标"；`用户的开发习惯？` 首位"偏好"（6/10 未命中） | `用户的目标岗位是什么方向？`、`用户在哪个城市？`、`用户对什么感兴趣？`、`用户有什么使用偏好？` 四条 top1 直接命中 |
| top3 命中 | `用户用什么工具部署？` top5 = [目标, 学习, 位置, 工具, 工具]，工具第 4 位才出现；`用户的开发习惯？` top5 = [偏好, 偏好, 兴趣, 目标, 习惯]，习惯第 5 位 | 其余 8 条 top3 内命中（含技术栈、项目、学习方向、职业背景） |
| 写入延迟 | 单轮 max 6.02 ms（第 2 轮）；p95 3.01 ms。绝对值仍低，属抖动不是故障 | 三轮 avg 均 ≤1.83 ms；首条冷启动已由 warmup 排除 |
| 检索延迟 | 第 1 轮出现 max 50.56 ms 的单点尾延迟（p95 仅 3.72 ms），疑似 Redis/网络瞬时抖动 | p50 1.17 ms、p95 2.09 ms；会话画像读 p95 1.66 ms |
| 去重 | 第 1 轮 3 写 0 跳、第 3 轮 2 写 2 跳：仅完全重复被跳过，近义事实仍重复入库 | 第 2 轮 3 条全部跳过（0 写 3 跳），说明精确重复路径有效 |
| 过期/降级 | 无失败 case（过期记忆未出现、未降级） | 两条断言均通过，可作为生命周期的高指标证据 |

### 4.3 根因判断

- **类别语义混淆。** query"工具部署"的 top5 前三位是目标、学习、位置，工具事实排到第 4；"开发习惯"与两条"偏好"事实共享"习惯/喜欢"措辞，导致偏好重复占据前两位。top5 100% 说明事实都在库里，缺的是**类别感知排序**。
- **乘性时间衰减（MEMORY_REWEIGHT=mult）**会放大近期/高频条目的权重，旧事实被压后；测试事实均为同一批次写入，衰减不应造成如此大的类别漂移，主因仍是语义相似度。
- **重复类别占位。** "使用偏好" top5 出现两条"偏好"，说明同一类别多条事实互相挤占名额，top1 命中但信息冗余。

---

## 五、检索缓存层

### 5.1 指标总览

| 指标 | 口径 | 结果 |
|---|---|---|
| 同义改写命中率 | 09-19，12 组×3 改写 = 36 | 35/36 = 0.9722；avg 554 ms / p95 2148 ms / max 3704 ms |
| 混合库存 correct 命中 | 09-19，库存 1/3/6/12，topk 3 或 12 | 80/102 = 78.4%；wrong 全为 0 |
| KNN 长列表召回 | 09-19，36 条，长列表 32 条 | recall@1 .3611 / @3 .6111 / @5 .6667 / @8 .8333 / @12 .8889；not_found 4 |
| 负样本误命中 | hard 8 + unrelated 6 | 0 / 0 |
| 缓存 embedding 缩减 | 09-19，24 条流式 query，width=4 | 4.0 → 3.5 文本/query，reduction 12.5%；HTTP 24 → 48，avg 1171.72 → 1191.24 ms |
| 灵敏度 | width 2/3/4/5 | 2 时 −25.0%，3 时 0，4 时 +12.5%，5 时 +20.0% |
| 精确重复命中 | 09-18 | 1.0 |
| 同义命中（旧） | 09-18，3 条 | 3/3 |
| TTL 策略（legacy） | 10 query，ttl 5 s | fixed 0/10 vs dynamic 10/30（33.3%），+33.3 pp |

### 5.2 逐指标 bad case / 高指标 case

| 指标 | bad case | 高指标 case |
|---|---|---|
| 同义命中率 | B07 组 2/3，是 12 组里唯一漏检；36 条中仅此 1 条未命中 | B01–B06、B08–B12 共 11 组 3/3；整体 97.2% |
| 混合库存命中 | `size12_topk3` 22/36 = 61.1%（14 条漏检）；`size6_topk3` 14/18 = 77.8% | `size1_topk3` 3/3、`size3_topk3` 9/9 = 100%；`size12_topk12` 32/36 = 88.9%（topk 放大救回 10 条） |
| KNN 长列表 | recall@1 仅 36.1%；rank6+ 8 条、not_found 4 条 | recall@12 88.9%；rank1 13 条、rank2–3 9 条 |
| 负样本 | 无失败（hard/unrelated 均 0 误命中） | 14 条负样本全部正确拒识，是该层最强指标 |
| 缓存收益 | width=2 时 2.5 vs 2.0 文本/query，reduction −25%；命中 50% 时 `embedded_texts` 96→84（−12.5%），但 HTTP 调用 24→48、avg 1171.72→1191.24 ms；省的是嵌入文本量，不是请求次数 | `isolated_size1` 场景 71.5% 缩减；size1/size3 命中 100% 时 75% 缩减 |
| 延迟 | `size6_topk3` max 120892 ms（单次异常，疑似在线重排接口超时，见报告 caveat 4），把 avg 拉到 7055 ms；p95 仍 594 ms | p95：size1 309 ms、size3 499 ms、size6 594 ms、size12 2902/3657 ms |
| TTL | fixed_ttl 0/10：5 s TTL 下 10 条顺序查询已全部过期，属构造性 0 分，不能读成"固定 TTL 失效" | dynamic_ttl 10/30 = 33.3%，相对固定 TTL +33.3 pp |

### 5.3 根因判断

- **库存规模稀释 topk。** 12 条库存、topk=3 时，正确 key 经常排在第 4–6 位；topk 提到 12 后命中回到 88.9%。这是"候选槽位不足"，不是相似度算错（wrong 始终为 0）。
- **KNN 首名弱。** 32 条长列表样本中只有 13 条排第 1，8 条排 6 名以后，4 条完全找不到；说明缓存索引对"同主题多条目"的区分度不足，需按 doc_id/thread 过滤或分层索引。
- **命中路径的 embedding 成本要分层看。** L3a 精确层在 `cache_nodes.py:46-50` 直接返回，命中即 0 embedding / 0 rerank；L3b 语义层在 `cache_service.py:565` 先 `embed_query_cached(query)` 再做 KNN，命中仍要付 query 向量（L2 嵌入缓存可抵扣一部分）。报告 caveat 第 2 条写的 `cache_service.py:202` 已过期（该行现为建索引），引用时需按当前代码位置修正。实测 24 条 query、50% 命中只省 12 次嵌入（12.5%），因此**语义缓存只在宽检索（width≥4）且命中率高时有净收益；width=2 时为负收益**。
- **命中路径的 HTTP 调用没有下降。** 50% 命中时嵌入文本 96→84，但 HTTP 24→48，平均延迟 1171.72→1191.24 ms；说明省下的是被嵌入的文档数，不是请求次数。需要查 `query_cache` 命中后是否仍走了 L3b 的嵌入/重排，以及 L2 嵌入缓存是否真正命中。
- **在线重排放大尾延迟。** 120.9 s 的单点超时与 caveat 第 4 条一致（SiliconFlow 在线接口抖动）；p95 与 max 差两个数量级，报告应同时给 p95 与 max。

---

## 六、MCP 工具层

### 6.1 指标总览

| 报告 | 样本量 | 结果 |
|---|---:|---|
| 工具筛选（09-18） | 22 条 / 41 工具 / top_k 12 | recall 0.8939；precision 0.1406（上限 0.1477）；zero_hit 0 |
| 工具装配（09-19） | 6 条 | 6/6 = 1.0 |
| 工具安全（09-19） | 11 条 | 11/11 = 1.0；command 2/2、package 2/2、sse 3/3、type 1/1 拦截率 1.0 |
| 工具截断（09-19） | 13 条 | 13/13 = 1.0；异常映射 3/3、描述截断 1/1、文档截断 1/1、轮次保护 4/4、熔断 4/4 |

### 6.2 逐指标 bad case / 高指标 case

| 指标 | bad case | 高指标 case |
|---|---|---|
| 筛选 recall | `看看有哪些文件被修改了` 0.50（期望 git_status+git_diff，只中 git_status）；`更新我的个人信息` 0.50（缺 create_relations）；`帮我分析这个项目的代码结构` 0.50（缺 directory_tree）；`读取配置文件并修改数据库连接` 0.50（缺 write_file）；`看看最近的代码改动并生成总结` 0.67（缺 read_file） | 17/22 条 recall 1.0，含 `读取 README`、`创建 architecture.md`、`列出目录文件`；zero_hit 0 |
| 筛选 precision | 最低 0.083（1 个正确 / 12 候选）；`搜索项目中包含 TODO 的文件` 混入 web_search；`查询用户表中所有用户` 混入 write_query/create_table（真实误选，非纯噪声） | 最高 0.25（3 个正确 / 12 候选）；17 条仍有 1–3 个正确工具，说明主因是候选池过大而非完全失效 |
| 装配 | 无失败 case | 规则命中、规则无命中返回空、语义并集去重、语义异常降级规则层、熔断跳过语义、两路未命中返回空，6 条全过 |
| 安全 | 无拒绝失败；`env_filter` 只有 1 条"敏感 env 过滤"用例，且是过滤而非拒绝（PATH/HOME 被删、API_KEY 保留），覆盖偏薄 | 非法命令、非白名单 npm 包、内网 localhost/127.0.0.1、ftp 协议、非法 type 全部拦截；合法配置与合法公网 sse 2 条放行 |
| 截断/熔断 | 无失败 case | ENOTDIR 能给出"path 必须是目录，请改用 read_file"的纠正方向；轮上限 8、新轮归零、轮内累计、防回退；连续失败 2 次熔断、失败 1 次不熔断、失败-成功-失败不熔断、节点注入提示不误判 |

### 6.3 关键判断

- **precision 0.1406 主要是候选太多，但不能全归因于噪声。** 期望工具平均 1.77 个、候选固定 12 个，算术上限 0.1477，实测已达 95.2%；但 `web_search`、`write_query`/`create_table` 是被误选进来的真实错误，说明语义层对"项目内搜索 vs 联网搜索""只读查询 vs 写查询"的边界不清。**结论：候选池过大是主因，动作边界误选是次因。**
- **漏检集中在"多工具组合"query。** 5 条 recall<1 的用例都要求 2–3 个工具协同（git_status+git_diff、read_file+write_file、add_observations+create_relations、get_project_info+directory_tree）；单工具 query 全部命中。说明规则/语义层对"动作链"的覆盖不足，而不是单工具识别失败。
- **机制层 30/30 全过是亮点，但覆盖面小。** 装配 6 条、安全 11 条、截断 13 条，均为构造用例；没有真实故障注入（如 MCP 服务端超时、返回畸形 JSON、工具描述缺失），也没有阈值调参记录（轮上限 8、熔断 2 次的取值依据）。

---

## 七、跨层结论与改进优先级

### 7.1 短板排序（按影响面与证据强度）

1. **RAG 召回（P0）。** 长代码块/参数清单类问题在切片后被拆散，相关块进不了 top-20；直接导致 correctness 0.20–0.40，且 after_rebuild 出现"新增内容仍 0 召回"的索引一致性问题。
2. **记忆排序（P1）。** top1 40%、两条 query top3 失败；事实都在库里，缺类别感知与同义扩展。
3. **缓存规模扩展（P1）。** size12/topk3 61.1%、KNN recall@1 36.1%；库存变大后候选槽位与索引区分度不足。
4. **工具候选过多（P2）。** recall 已 89.4%、zero_hit 0；precision 受 top_k=12 压制，但也存在 `web_search` vs `search_project_files`、`read_query` vs `write_query/create_table` 的动作边界误选，属配置、排序与规则约束问题。
5. **评估口径（P0，影响所有结论）。** judge 非官方库、上下文截断、21/28 混用、p95 异常、precision 上限，若不在报告中说明，数字会被质疑。

### 7.2 改进建议（可直接落地）

- **切片：** 把 Markdown 代码围栏作为硬边界，标题与正文绑定为父子块（small-to-big）；清单类问题召回父块，事实类问题召回子块。当前 500 字固定切分对 `add_node` 六行代码、`chunk_size=500` 参数表都不友好。
- **召回：** 对节点名、常量、函数名保留原始 query 做 BM25 精确匹配，与向量结果 RRF 融合；query 改写不要把 `add_node`、`chunk_size=500` 这类术语改写成自然语言。
- **索引一致性：** ingest 记录 content_hash 与批次号，重建后对新内容做 20 条抽检；本次 after_rebuild 的 Send 0.00 就是现成的回归用例。
- **评估：** judge 不截断或分段评分；同一题集跑 3 次取中位数并给区间；21 条与 28 条分开报告，禁止跨表比较。
- **记忆：** 检索前加 category 分类/过滤，对"工具""习惯"建立同义扩展；去重从精确匹配升级为语义相似度阈值（如 >0.9 合并）。
- **缓存：** 按库存规模自适应 topk（size12 用 topk12），按 doc_id/thread 建分层索引；width<4 时只保留 L3a 精确层、关闭 L3b 语义层，避免负收益。
- **工具：** top_k 12 → 5–6，或两阶段（规则召回 → LLM 精选）；对 `write_query`、`create_table` 等写操作加动作类型负向约束（查询类问题不得进入写工具），对"项目内搜索"与"联网搜索"加互斥规则；指标改用 precision@3 / NDCG / 动作链完整率，替代受上限压制的 precision@12。

### 7.3 可直接引用的亮点

- faithfulness 0.959（21 条）：correctness 最低的四条（切分、节点、密码、配置）faithfulness 均为 1.00，"不编造、缺资料就拒答"的引用约束有效；真正拉低该指标的是 SSE 的通识句（0.75）与 MCP/会话归属（0.80）。
- 缓存负样本 14 条 0 误命中；同义改写 97.2%。
- 记忆 top5 100%、写 1.77 ms / 读 1.34 ms、过期与降级断言全过。
- 工具机制层 30/30：装配降级、安全拦截、异常映射、轮次保护、熔断全部有构造用例并通过。

---

## 八、附：指标速查表

| 层 | 指标 | 数值 | 样本 | 主要 bad case | 主要高指标 case |
|---|---|---:|---:|---|---|
| RAGAS | context_precision | .6381 | 21 | 切分 .25、节点 .30 | 配置/密码 1.00 |
| RAGAS | context_recall | .8005 | 21 | 切分 .13、节点 .20、密码 .25 | 401/异常/JWT 1.00 |
| RAGAS | faithfulness | .9590 | 21 | SSE .75、MCP/会话归属 .80 | 15 条 1.00、18 条 ≥0.9 |
| RAGAS | answer_relevancy | .9881 | 21 | 切分 .80 | 多数 1.00 |
| RAGAS | answer_correctness | .7976 | 21 | 节点/切分 .20、密码 .25 | 401/SSE/异常 1.00 |
| 记忆 | top1/top3/top5 | .40/.80/1.00 | 10 | 工具 top3 失败、习惯 top3 失败 | 目标/位置/兴趣/偏好 top1 |
| 记忆 | 写/读延迟 | 1.768/1.341 ms | 240/900 | 读 max 50.56 ms（单点） | p50 1.57/1.17 ms |
| 缓存 | 同义命中 | .9722 | 36 | B07 2/3 | 11 组 3/3 |
| 缓存 | 混合 correct | 80/102 | 102 | size12_topk3 61.1% | size1/3 100%、size12_topk12 88.9% |
| 缓存 | KNN recall@1/@12 | .3611/.8889 | 36 | not_found 4 | rank1 13 条 |
| 缓存 | 负样本误命中 | 0/14 | 14 | 无 | 全部拒识 |
| 工具 | recall / precision | .8939/.1406 | 22 | 4 条 0.50、1 条 0.67 | 17 条 1.00；precision 上限 .1477 |
| 工具 | 装配/安全/截断 | 1.0/1.0/1.0 | 6/11/13 | 无 | 30/30 全过 |

---

## 九、复现命令与引用规范

```bash
cd src
python -m agent_test.eval_ragas_judge --dataset resources/knowledge-base/test-qa/eval_project_dataset.json --limit 21
python -m agent_test.eval_memory
python -m agent_test.eval_cache_hitrate
python -m agent_test.eval_tool_filter
```

引用规范（`reports/README.md` 要求）：对外写数字时必须带"日期 + 脚本名 + 样本量"，例如"2026-09-19，`eval_ragas_judge.py`，n=21，context_recall 0.8005"。不得只写文件名；21 条与 28 条可以并列展示，但必须各自标注 n 与题集，不能当作同一题集的分数直接比较。
