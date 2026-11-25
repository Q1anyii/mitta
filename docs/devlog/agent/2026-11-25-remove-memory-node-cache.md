# memory_node 节点级缓存移除：避免跨用户误命中导致记忆写入丢失

日期：2026-11-25

docs_sync: required

## 背景

`main_graph` 原先给 `memory_node` 配置了 LangGraph 节点级 `CachePolicy`：

```python
CachePolicy(ttl=CACHE_MEMORY_NODE_TTL, key_func=_memory_cache_key)
```

设计意图是“同一个节点输入在 TTL 内重复到达时，跳过节点执行，省一次 mem0 提取调用”。
经源码核对与最小图实测，这个缓存的收益和风险不匹配：

1. **缓存的是节点 writes，不是完整超步状态。** `memory_node` 返回 `None`，节点本身不修改
   state；LangGraph 仍会缓存一个 `__no_writes__` 哨兵。命中时节点函数不执行，效果只是跳过
   `mem0.add` 这个副作用，不是“缓存了记忆内容”。
2. **key 没有用户/会话作用域。** `_memory_cache_key` 只返回
   `len(messages)|input_str|last_ai`，缓存 namespace 是
   `('__pregel_ns_writes', '__dynamic__', 'memory_node')`，不含 `user_id` / `thread_id`。
   两个用户只要消息条数、输入、最后一条 AI 内容相同，10 秒内就会命中同一条缓存，
   第二个用户的 `memory_node` 被跳过，长期记忆不写。
3. **正常一轮只执行一次。** 主图没有工具回环到 `memory_node`；同一用户新回合的
   `len(messages)` 也会变化，基本不会命中。它防不住“用户重复问”，只在同一 checkpoint
   重放、重复投递或跨用户/跨线程撞 key 时生效。
4. **key 本身不稳定。** `last_ai` 是模型生成内容，模型输出一变就不命中；`|` 无转义，
   `len` 不是内容指纹，存在拼接碰撞。
5. `memory_node` 已经 fire-and-forget，缓存省的不是 SSE 延迟；后台 `mem0.add` 失败时，
   缓存条目还可能已写入 `__no_writes__`，10 秒内重试会被跳过。

用户确认：该节点在收尾阶段、工作流无循环，重复到达场景基本不可能，直接删除缓存。

## 改动

1. `src/graphs/main_graph.py`
   - 删除 `CachePolicy` 导入；
   - `memory_node` 不再传 `cache_policy`，只保留 `add_node("memory_node", memory_node_bound)`；
   - 删除 `_memory_cache_key` / `CACHE_MEMORY_NODE_TTL` 导入；
   - 更新注释：说明节点无 state 写入、缓存无法按 user/thread 隔离、重复事实交给 mem0
     hash 去重。
2. `src/graphs/nodes/memory_node.py`
   - 删除 `_memory_cache_key` 函数；
   - 删除 `uuid` / `AIMessage` 死 import；
   - 模块 docstring 改为“不做节点级缓存，重复事实由 mem0 hash 兜底”。
3. `src/constant/cache_constant.py`
   - 删除 `CACHE_MEMORY_NODE_TTL = 10` 及其注释。
4. `tests/test_agent_regression.py`
   - 删除 `TestMemoryCacheKey` 的 3 个用例、`_memory_cache_key` 导入与 `ToolMessage` 死 import；
   - 更新文件头覆盖范围说明。

## 验证

- `ast.parse`：`main_graph.py` / `memory_node.py` / `cache_constant.py` / `test_agent_regression.py` 通过。
- `pytest tests/test_agent_regression.py -q`：16 passed。
- `pytest tests -q`（`LLM_BACKEND=mock` / `SESSION_STORE=memory` / `LOG_STORE=memory`）：
  70 passed, 1 warning（LangGraph 序列化 pending deprecation warning）。
- Grep 确认生产代码中已无 `CACHE_MEMORY_NODE_TTL` / `_memory_cache_key` /
  `cache_policy=CachePolicy` 引用；仅 `src/agent_test/test_graph.py` 的测试图仍使用
  `CachePolicy`，与本节点无关。

## 文档同步项（docs_sync: required）

以下文档仍描述 memory_node 缓存，需文档撰写 Agent 同步：

- `README.md:59`：技术栈表仍列 `CachePolicy`，当前生产主图已无节点级 CachePolicy。
- `docs/FEATURES.md:24`：仍写“memory_node 结果按 TTL 缓存”。
- `docs/DESIGN_NOTES.md:56-69`：整节“节点级缓存”仍以 memory_node 为唯一示例，需改为
  “生产主图已移除节点级缓存”或删除该节，并保留 retrieve/tool 缓存移除原因。
- `docs/architecture-flowcharts.md:16` 的 tool_node CachePolicy 描述与当前代码也不一致
  （tool_node 明确未启用缓存），可一并核对。

## 遗留

- 本次未改记忆写入策略；`mem0.add` 的 hash 去重仍是完全重复文本的兜底，语义重复不自动
  update/delete。
- Mitta 处于冻结期，本次仅移除缓存，不改其他记忆链路。
