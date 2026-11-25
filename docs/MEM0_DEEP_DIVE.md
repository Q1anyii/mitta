# Mem0 深度钻研笔记（Mitta 长期记忆改造）

> 配套源码：`mem0 2.2.1`（本地 `site-packages/mem0`）+
> Mitta 封装层 `src/service/memory_service.py` + 评测 `src/agent_test/eval_memory.py`。
> 时间：2026-10-29 整理（P1–P6 改造完成后）。

---

## 1. 写入管线（`Memory.add`，V3 分阶段 8 阶段）

| 阶段 | 做什么 | 关键点 |
|---|---|---|
| P0 上下文收集 | `db.get_last_messages(limit=10)`（**sqlite 历史**最近 10 条，DESC 取再 ASC 排）+ `parse_messages(本轮 messages)` | 两个"10"之一：**时间最近 10 条消息** |
| P1 已有记忆检索 | 本轮消息文本 embed → `vector_store.search(top_k=10)` → UUID 映射防幻觉 | 两个"10"之二：**语义最相似 10 条记忆**（不是时间最近！） |
| P2 LLM 提取 | 单次调用：`ADDITIVE_EXTRACTION_PROMPT` + `generate_additive_extraction_prompt(existing_memories, new_messages, last_k_messages, custom_instructions)`；`response_format=json_object`，容错 `remove_code_blocks/extract_json` | **ADD-only**（只产生 add 操作）；合并/增量决策下沉到 prompt |
| P3 批量 embed | `embed_batch` 全部事实一次向量化，失败逐条 fallback | — |
| P4/P5 hash 去重 | `md5(text)` 撞已有/本批 hash 即跳过 | **重复事实不重复入库** |
| P6 批量插入 | 一次 insert，失败逐条 fallback | 落库才记历史 |
| P7 实体链接 | 抽实体 → entity + `linked_memory_ids` | 供检索 entity boost |
| P8 保存消息 | `save_messages(当前轮消息)` 入库 | 下次 add 的 P0 `last_messages` 来源 |

**parse_messages 只认 system/user/assistant 三种 role 且有 content 的消息；`role=tool` 直接不输出，assistant 纯 tool_calls 无 content 被跳过** → 工具调用内容不进提取上下文，AI 正文回复（llmout）会进。

**"20 条"真相**：existing（语义 top10）+ last_messages（时间最近 10 条）都进**提取 prompt**（非对话上下文）。OSS 硬编码 top_k=10 / limit=10，不可配。相对 store 旧方案"全量档案随轮数线性膨胀"，20 条是**固定成本上界**，且在我们封装里后台异步、不阻塞响应。

---

## 2. 检索管线（`_search_vector_store`，9 步）

1. 预处理：`lemmatize_for_bm25(query)` + `extract_entities(query)`
2. embed query
3. 语义检索 over-fetch：`internal_limit = max(limit*4, 60)`
4. BM25 关键词检索（同量级；redis 后端 = RediSearch `TextQuery(text_field_name="memory")`，返回 `text_score`）
5. BM25 sigmoid 归一化（见 §3）
6. entity boost（query 有实体时 `_compute_entity_boosts`）
7. 候选集：从语义结果过滤过期（`_payload_is_expired`：`expiration_date < 今天(UTC)` 即隐藏；`show_expired=True` 才见）
8. `score_and_rank`（threshold 门控 + 融合 + 归一化）
9. 格式化（promote 关键 payload key）

> 本质 = RAG 混合检索（向量 + BM25 多路召回 + 融合排序），多了**过期过滤**和**threshold 门控**两个 RAG 通常没有的维度。

---

## 3. 评分公式与 `max_possible` 设置（`mem0/utils/scoring.py`）

### 3.1 三路信号
- **semantic**：redis 后端 = `max(0, 1 - vector_distance)`（有界 [0,1]）
- **BM25**：原始分 0–20+（**随 query 变长系统性偏高**），需归一化
- **entity**：固定权重加分 `ENTITY_BOOST_WEIGHT = 0.5`

### 3.2 threshold 门控（融合前！）
```
if semantic_score < threshold: continue   # 直接排除，BM25/entity 加分救不回
```
门控看的是**纯语义分**，在 score_and_rank 内部、融合之前（scoring.py:110-112）。

### 3.3 BM25 归一化（sigmoid）
```
normalize_bm25 = 1 / (1 + e^(-steepness * (raw - midpoint)))   # ∈ [0,1]
```
参数按 lemmatized query 词数自适应（`get_bm25_params`）：

| 词数 | midpoint | steepness |
|---|---|---|
| ≤3 | 5.0 | 0.7 |
| ≤6 | 7.0 | 0.6 |
| ≤9 | 9.0 | 0.5 |
| ≤15 | 10.0 | 0.5 |
| >15 | 12.0 | 0.5 |

### 3.4 max_possible 怎么设置（scoring.py:94-101）
```
max_possible = 1.0            # semantic 必有
if has_bm25:  max_possible += 1.0    # +1.0
if has_entity: max_possible += 0.5    # +ENTITY_BOOST_WEIGHT
```
| 激活信号 | max_possible |
|---|---|
| 仅语义 | 1.0 |
| 语义 + BM25 | 2.0 |
| 语义 + BM25 + entity | 2.5 |
| 语义 + entity | 1.5 |

### 3.5 最终分
```
combined = min((semantic + bm25 + entity_boost) / max_possible, 1.0)
```
按 combined 降序取 top_k。

**归一化就是防止高挤占低吗？** 是。BM25 原始分 0–20+ 且长 query 更高，不压到 [0,1] 会把 0–1 的语义分完全淹没；sigmoid 后三路信号同量级可加，max_possible 除回去保证有界，entity 权重 0.5 是设计上的降权。注意 semantic 本来就归一，真正需要压的是 BM25。

> 口径区分：mem0 是**分数级加权线性融合**，不是 RRF（秩融合 `Σ1/(k+rank)`）。不用 RRF 的原因：threshold 门控依赖 semantic 连续分（1−distance），RRF 抹平分数会让门控失效。

---

## 4. reweight 双模式（`memory_service.py` 封装层，P5）

### 4.1 常量
```
_RECENCY_WINDOW_DAYS = 30            # 新近窗口
_ACCESS_WINDOW_DAYS  = 7             # 访问窗口
_W_RECENCY = 0.15                    # add 模式 recency 权重
_W_ACCESS  = 0.25                    # add 模式 access 权重
_W_TIME    = 0.8                     # mult 模式强度（time_factor ∈ [0.6, 1.4]）
_ACCESS_KEY_TPL   = "mem0:access:{}" # redis hash：mem_id → last_access
_ACCESS_TTL_SECONDS = 7 * 86400      # hash 整体 7 天 TTL
MEMORY_REWEIGHT：env 二选一，默认 "mult"（"add" 为回退）
```

### 4.2 时间信号（每条命中记忆）
```
recency_signal = max(0, 1 - age_days/30)            # ∈[0,1]，30 天内越新越高
access_signal  = max(0, 1 - ago_days/7)（无记录=0） # ∈[0,1]，7 天内被检索过越高
```

### 4.3 add 模式（加分，回退路径）
```
bonus = 0.15 * recency_signal + 0.25 * access_signal
score = raw + bonus          # 微调排序，不反转语义主序
```

### 4.4 mult 模式（乘法衰减，当前默认）
```
freshness   = (recency_signal + access_signal) / 2
time_factor = 1 + 0.8 * (freshness - 0.5)           # ∈ [0.6, 1.4]
score       = raw * time_factor                     # 可反转语义主序
```
刚访问的新记忆 ×1.4 放大，久未访问的旧记忆 ×0.6 压低。
**本质区别**：乘法改变量级（0.85×0.6=0.51 掉出前三），加法只平移（0.85+0.4=1.25 稳居第一）——mult 是"时间成为重排主力"。

### 4.5 位置与副作用
- 在 **mem0 `score_and_rank` 之后**执行，只重排"过了 threshold 门"的候选；门控仍看原始语义分，reweight 不污染门控判定。
- 访问计数：fire-and-forget `HSET mem0:access:{uid} {mem_id} {now}` + 7 天 TTL；redis 挂了降级为仅 recency。
- 输出：`raw_score` 保留原分，`score` 为加权后分值，降序返回。

---

## 5. 封装取舍（`memory_service.py` 设计决策）

| 层 | 取舍 | 代价（如果不做） |
|---|---|---|
| 懒加载 + 全方法降级 | mem0 未装/失败返回 None/空，add/search 异常记日志 | 生产被记忆库绑架，一个 bug 带崩对话 |
| 双后端 env 切换 | chroma 本地零依赖 / redis（需 redisvl+RediSearch，**FT 索引只能建 db 0** → 独立实例隔离，compose `redis-memory` noeviction 防驱逐） | 单后端绑死部署形态 |
| `_patch_mem0_redis_compat` 运行时补丁 | `setattr(SearchIndex, "set_client", 兼容实现)` 桥接 mem0 2.2.1 与 redisvl 0.28.0 API 差异；**不 fork、不改第三方文件**，初始化前幂等执行 | 降 redisvl 伤业务缓存/限流 |
| `_DEFAULT_EXTRACT_PROMPT` 强制原文语言 | 修 DeepSeek 把中文事实翻英文的实测 bug（中文 BM25 失配，P6 首轮 3/10 miss 根因） | 中文记忆检索质量崩 |
| `expiration_days` 包装 | 默认 90 天 TTL（`MEMORY_TTL_DAYS` 覆盖，0 关闭）；mem0 原生要调用方自己算日期 | 调用方到处重复 |
| `_reweight` 双模式 + 访问计数 | OSS 无 `reference_date`（平台功能，传参 ValueError），时间维度自管；访问计数给"被检索→权重提高"的 LRU 语义 | 纯语义排序，过时记忆永远压过新记忆 |
| 进程内单例 | `get_memory_service` + lock，llm_node/memory_node 共用 | chroma 重复初始化、多连接 |

主线：**降级优先、版本兼容隔离、OSS 缺的能力（时间维度/语言修复/过期包装）补在薄封装层，不改上游。**

---

## 6. 评测依据（`src/agent_test/eval_memory.py`）

- **为什么重点在检索质量**：读写就一次 redis IO，测延迟没意义（`--include-latency` 可选，默认不跑）。
- 参数：`FACT_POOL` 15 条事实带 `metadata.category`（= 答案）；`RETRIEVAL_QUERIES` 10 条 query → 期望类别；**`threshold=0.5` / `top_k=5` 与生产 llm_node 完全一致**。
- 指标：top1/top3/top5 命中率 = 期望类别出现在结果前 1/3/5 位比例（**recall@k 变体**，category 自动断言，免 LLM judge）；生命周期用例（昨天过期记忆隐藏 ✓ / 永久可见 ✓）。
- 结果（threshold 0.5 + mult）：**top1 40% / top3 80% / top5 100%**；0.3→0.5 让 top5 从 80% 升到 100%。
  - top1 偏低根因：高分泛化事实互相挤压（职业/目标/兴趣都含"AI 应用开发"争第一）+ 单次 LLM 提取波动，不是检索失效。
  - mult vs add：批量同批 add 数据时间无差异，两者结果一致；针对性排序验证——40 天前 raw 0.85 在 mult ×0.6=0.51 掉出前三，1 天前 0.55 反超 → 时间成为重排主力。

---

## 7. 关键结论速记（便于复查）

1. **为什么从"每轮全量合并画像"改 mem0？** 全量成本 = 档案长度 × 轮数线性膨胀；mem0 增量提取固定成本、只写新事实、md5 去重。
2. **两个"十条"**：existing = 语义检索 top10（不是最近）；last_messages = 时间最近 10 条历史消息。都只进提取 prompt。
3. **threshold 挡什么分**：纯语义分（redis = 1−距离），融合前排除；BM25/实体救不回。
4. **为什么不是 RRF**：门控依赖语义连续分；分数级加权和 > 秩融合信息量。
5. **时间加权怎么做**：recency 30 天 + access 7 天双信号，mult 模式 `score = raw × (1 + 0.8×(freshness−0.5))`，90 天 TTL 过期隐藏。
6. **评测怎么测**：10 条中文 query 按 category 自动断言，top5 100%（参数与生产一致）；含过期隐藏/长期可见用例。
