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

## 遗留

- P5（生命周期/时间衰减，OSS 受限只能 created_at 手动）、P6（评测/灰度）未做
- 过时注释待文档侧同步：`graphs/utils/per_user_lock.py`、`graphs/utils/user_profile.py`、
  `agent_test/eval_memory.py` 仍描述旧 store 合并语义
- 提交记录：P1+P2（c4192a5）、requirements mem0ai（bb9c214）、P3（53acc91）；
  P4 本次提交；均未推送
