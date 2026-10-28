# mem0 记忆改造（P2 写路径 + P3 读路径 + P4 redis 后端）：store 全量合并 → mem0 增量入库 + 检索注入

日期：2026-10-27
docs_sync: required

## 背景

长期记忆原方案：memory_node 每轮 `store.get(user_profile)` → LLM 全量合并 → `store.put`。
痛点：LLM 输入 = 全量档案，随记忆增长线性膨胀；每轮一次合并调用，token 成本 = 档案长度 × 轮数。

改造方向（用户拍板）：mem0 检索式记忆——增量事实提取入库 + 向量检索注入。
P2 只动写路径：memory_node 不再做全量合并，改为把本轮对话提交 `memory_service.add`。

## 改动

1. **`src/graphs/nodes/memory_node.py`**（重写收尾）
   - 签名 `(state, config, store, model)` → `(state, config, memory_service)`，删除 store/model 依赖
   - 删除函数体内自建的 `MemoryService()` 实例（与注入矛盾）
   - `_extract_and_persist` 整段（store.get → LLM 合并 → store.put）替换为 `_add_memory`：
     `messages = [{"role":"user","content":input_str}, {"role":"assistant","content":ai_reply}]`
     → `memory_service.add(uid, messages)`
   - 保留：快速路径（idle/unavailable 且无需检索跳过）、fire-and-forget（executor.submit）、
     `_memory_cache_key` 缓存逻辑（逻辑未变，docstring 由"store 写入"更新为"mem0 写入"）
   - 清理废弃 imports：re / HumanMessage / BaseStore / MEMORY_EXTRACT_PROMPT / NO_INFO_MARKS /
     get_per_user_lock / MemoryService / threading

2. **`src/graphs/main_graph.py`**
   - 顶部 import `get_memory_service`
   - `partial(memory_node, memory_service=memory_service)` 前补 `memory_service = get_memory_service()`
     （进程内单例；mem0 缺失/失败返回 None，节点内部降级）——修复原写法 NameError
   - 更新三处过时注释（文件头 docstring、memory_node 注册注释：store.put → mem0.add）

## 设计决策

- **store 的 user_profile 职责移交**：`_ensure_username_profile`（llm_node 回答前同步）已独占维护
  用户名档案（低频，仅用户名缺失/变更时写）。memory_node 的 get/put 整体删除后该档案不受影响，
  "保留 store 就得每轮 get 再 put" 的矛盾不复存在。
- **per_user_lock 移除**：原锁防"读→合并→写回"的 last-write-wins 竞态，随合并删除而消失；
  mem0 内部对重复事实有 hash 去重。user_profile.py 内自用锁保留（单写者，无害）。
- **成本**：每轮 = 本轮对话（固定量）+ 一次 bge-m3 embedding；mem0 仅检测到重复事实时额外
  调一次 LLM 合并。token 从"档案长度 × 轮数"收敛为"对话长度 × 轮数"。

## 验证

- conda 环境（无 mem0）：
  - AST parse 两文件通过
  - `import graphs.main_graph` 通过（get_memory_service 已加载）
  - memory_node 行为测试：idle 轮快速路径跳过（不调 add）；executed 轮正确组装
    `[user, assistant]` 调 add；`_memory_cache_key` executed 轮确定性 key / idle 轮 nocache
- Python 3.13（有 mem0）服务层 add 已在此前 P1 验证调通（本节点只包装调用）

## P3 读路径（追加于同日）

- `src/graphs/nodes/llm_node.py`：签名加 `memory_service=None`（默认降级）；`_ensure_username_profile`
  后接 `memory_service.search(query=input_str, user_id=user_id, top_k=5, threshold=0.3)`，
  结果拼 `【用户近期记忆】` 块注入 system prompt（与 `【用户长期记忆】` 并存）。
- `src/graphs/main_graph.py`：`memory_service = get_memory_service()` 单例前移至 llm_node_bound
  前，`llm_node` 与 `memory_node` 共用同一单例。
- 设计：**两层记忆并存，非替代**——store（PG）稳定档案（用户名，每轮必带，低频写）
  + mem0（向量库）高频事实（按需检索注入）。store 方案取消的仅是"每轮 LLM 全量合并写入"，
  读路径与用户名低频写保留；长期事实增量来源移交 mem0。
- 验证（conda，mock 依赖）：`mem_injected=True`（检索结果正确注入）、`none_ok=True`
  （memory_service=None 降级不炸）；AST 通过。

## P4 redis 后端切换（追加于同日）

- 背景：默认 chroma（本地文件库）单机持久化、零依赖，但部署可观测性弱、无 BM25；
  redis 后端（redis-stack 含 RedisSearch）可观测、可检索、BM25 原生。
- **发现一：redis 后端 BM25 是原生能力，无需插件**。mem0 2.2.1 `Memory.search` →
  `_search_vector_store` 无条件调用 `vector_store.keyword_search()`（BM25 向量混合检索，
  over-fetch internal_limit=max(top_k*4, 60)）。用户此前设想的"把 BM25 放 redis 的插件"
  不必写代码。
- **发现二：mem0 2.2.1 与 redisvl 0.28.0 API 不兼容（上游 bug）**。RedisDB.__init__ 调
  `SearchIndex.from_dict(schema)` + `index.set_client(client)`；redisvl>=0.3 已移除
  set_client（改构造参数 redis_client，内部属性 `_SearchIndex__redis_client`）。
  同时 mem0 METADATA 约束 `redis<6.0.0,>=5.0.0` 与项目 `redis==8.1.0`（业务缓存/BM25/限流
  依赖）冲突——**不能降 redis**。
  方案：`memory_service._patch_mem0_redis_compat()` 运行时补丁——给 SearchIndex 补
  `set_client(client)`（等价注入 redis_client），随项目代码部署，不 patch 第三方文件。
  补丁幂等；验证：set_client_patched=True、错误从 AttributeError 变为连接错误（API 已桥接）。
- **发现三：reference_date（时间衰减）是 mem0 平台功能**，OSS 版 search 传参会直接
  `raise ValueError`——P5 时间加权在自托管受限，只能按 created_at 手动排序。
- 部署隔离：业务 redis 是 `allkeys-lru`（256MB，LRU 逐低频 key），mem0 记忆不可重建 →
  docker-compose 新增独立 `redis-memory` 实例（redis-stack:7.2.0-v10，127.0.0.1:6380:6379，
  `--appendonly yes --maxmemory 256mb --maxmemory-policy noeviction`，卷 redis_memory_data），
  api 注入 `MEMORY_VECTOR_STORE=${...:-chroma}` / `MEMORY_REDIS_URL=${...:-redis://redis-memory:6379/0}`。
  db 隔离不防 LRU（实例级策略），故必须独立实例而非仅独立 db。
- requirements：新增 `redisvl==0.28.0`（redis 后端专用；默认 chroma 不需要）。
- 验证：`docker compose config --quiet` EXIT=0；Python 3.13 redis 配置构造正确
  （backend=redis / redis_url=redis://localhost:6379/1 / dims=1024 / collection=mem0）；
  无 redis 连接时初始化降级 [] 不炸；redisvl 0.28.0 import OK。
- **待办（未验证项）**：本地无可用 redis-stack（127.0.0.1:6379 握手失败非真 redis；
  docker daemon 未运行），全链路 add→search 实测待环境就绪（起 redis-memory 容器或
  提供可用 redis-stack 实例）。

## P4 实测（2026-10-28 追加）

- **环境打通**：WSL（Ubuntu）里跑着 Sorts 项目的 docker 容器，6379 是
  `sorts-redis`（redis/redis-stack-server，host 网络，`REDIS_ARGS=--requirepass sorts_dev
  --appendonly yes`，**无 maxmemory/LRU**）。Windows 侧经 WSL2 localhost 转发
  `redis://:sorts_dev@127.0.0.1:6379/0` 连接成功，FT._LIST 确认 RediSearch 可用。
- **实测发现的关键限制：RediSearch FT 索引只能在 db 0 创建**（`Cannot create index on
  db != 0`）。P1 默认值 `redis://localhost:6379/1` 是错的——已修为默认 db 0；
  隔离只能靠独立实例（compose `redis://redis-memory:6379/0` 恰好正确），
  "独立 db 不防 LRU"的结论升级为"独立 db 直接建不了索引"。
- **全链路实测通过**（Python 3.13 + 真实 DeepSeek/SiliconFlow key + sorts 6379 db 0）：
  - add ×4 全 True（首次 92s 含建索引+LLM 事实提取+embedding，后续 ~1.5s）
  - search 精准命中：名字 0.67 / 跑步 0.58 / 技术栈 0.63——**BM25+向量混合检索生效**
  - 用户隔离正确：u2 检索只返回 u2 记忆，u1 无泄漏
  - get_all 完整取回；mem0 2.2.1 初始化、set_client 补丁在真实后端工作正常
  - 实测数据已清理（mem0 reset + drop index，Sorts redis 恢复原状 ft_list=[]）
- 备注：add 首次慢（92s）是建索引 + LLM 提取 + embedding 串行；生产每次新用户首轮
  也会有此成本，后续轮次 ~1.5s 可接受。spaCy 未装不影响（BM25 lemmatize 有 fallback）。

## 遗留（今日待办，2026-10-28）

### 待办 1：redis 后端全链路实测 —— ✅ 已完成（见上"P4 实测"）
- 实测通过（add/search/隔离/get_all 全绿），db 0 限制已修复并需提交。

### 待办 2：P5 生命周期参数 —— ✅ 已完成（2026-10-28）
- 过期清理：mem0 `add` 原生支持 `expiration_date`（YYYY-MM-DD，search/get_all 自动隐藏
  过期）。`memory_service.add` 新增 `expiration_days` 参数，默认 `MEMORY_TTL_DAYS`（未设=90），
  传 0/负数显式关闭。
- 手动时间加权（替代 OSS 不可用的 reference_date）：`search` 结果按
  `加权分 = 原始分 + recency加分 + access加分(LRU)` 重排：
  - recency：created_at 距今 ≤30 天线性加分（越新越高，权重 0.15）
  - access：命中的记忆写 last_access 到业务 redis（`mem0:access:{user_id}` hash），
    距上次访问 ≤7 天线性加分（权重 0.25）——用户"记忆被访问权重提高"的 LRU 语义；
    redis 不可用自动降级为仅 recency。
  - 结果保留 `raw_score` 与 `id`。
- 验证（conda mock）：recency 加分精确（1天→0.145、15天→0.075、>30天→0）；
  access 加分精确（刚访问→0.25）；排序按加权分；TTL 默认90/显式30/0关闭全对；
  redis 不可用降级不炸；reference_date 默认不传。

### 待办 3：P6 评测 / 灰度 —— ✅ 已完成（2026-10-28）
- **评测脚本重写**（`agent_test/eval_memory.py`）：PostgresStore 合并语义 → mem0 检索式。
  按用户意见"评测重点在检索质量而非读写耗时"（读写就是一次 redis IO），默认只跑
  **检索质量 + 生命周期**；延迟测试保留为 `--include-latency` 可选项。
  - 检索质量：add 时带 `metadata.category`，按 query→预期类别自动断言 top1/top3/top5
    命中率；**口径对齐生产**（llm_node 注入 top_k=5, threshold=0.3）。
  - 生命周期：已过期记忆（昨天过期）不出现、未过期可见。
- **修复一：search 丢 metadata**。memory_service.search 原只返回 id/text/score/created_at，
  评测拿不到 category（首轮命中率 0% 是脚本问题，非检索问题）。search 返回补 metadata。
- **修复二（真实质量问题）：提取记忆变英文**。mem0 默认 FACT_RETRIEVAL_PROMPT 虽带
  "record in the same language"，实测 DeepSeek 仍把中文事实翻成英文 → 中文 query 检索时
  BM25 失配、向量跨语言损耗（首轮 3/10 miss 的根因）。修复：memory_service.add 默认传
  `_DEFAULT_EXTRACT_PROMPT`（强制保留原文语言，保持 {"facts": [...]} JSON 契约）。
- **修复三：评测过期用例 bug**。原用 expiration_days=1（明天才过期）测"隐藏"必然失败；
  改底层调用写"昨天过期"的记忆验证隐藏。
- **本地实测结果**（Python 3.13 + 真实 key + Sorts redis 6379 db 0，threshold=0.3）：
  - 检索质量：**top1 50% / top3 70% / top5 80%**（10 题；提取语言修复后中文 query 命中
    显著回升，位置/技能 miss 修复）
  - 生命周期：已过期隐藏 ✓ / 未过期可见 ✓
  - 剩余 2/10 miss（项目/职业）：高分泛化事实（职业/目标/兴趣均含"AI 应用开发"）挤压
    top5 位置 + 单次 LLM 提取波动；检索本身正确（小场景 3 条记忆时全部命中）。
  - 对比旧方案：PostgresStore 无检索（每轮全量画像塞给 LLM），mem0 检索式 80% top5
    召回 + 过期清理 = "有 RAG vs 没 RAG"的差距，**支持同步生产**。
- **灰度方案**：代码已就绪（P2/P3/P4/P5 全绿 + 评测达标）。生产切换 = 部署时
  `MEMORY_VECTOR_STORE=redis` + 起 compose `redis-memory`（独立实例、noeviction）+ 
  `MEMORY_REDIS_URL=redis://redis-memory:6379/0`；降级路径已验证（redis 不可用自动降级
  空检索，主链路不受影响），切换风险可控。建议随下一部署窗口切换，先小流量观察。
- 报告：`src/agent_test/memory_eval_report.json`（含 10 条 query 的 top5 类别明细）。
- 提交：c2b012e 后新增提交（P6 评测重写 + 语言约束 + search metadata 修复）。

### 已知遗留（无需动作，文档侧同步）
- 过时注释待文档同步：`graphs/utils/per_user_lock.py`、`graphs/utils/user_profile.py`
  仍描述旧 store 合并语义
- 提交记录：P1+P2（c4192a5）、requirements mem0ai（bb9c214）、P3（53acc91）、
  P4（51ef376）、P4 修复（87f0461）、P5（c2b012e）；均未推送
