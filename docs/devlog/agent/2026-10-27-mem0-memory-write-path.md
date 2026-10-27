# mem0 记忆写路径改造（P2）：store 全量合并 → mem0.add

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

## 遗留

- P3 读路径未做：llm_node 注入 `memory_service.search(user_id, input_str)` 检索记忆
- P4（redis 后端切换）、P5（生命周期参数）、P6（评测/灰度）未做
- 过时注释待文档侧同步：`graphs/utils/per_user_lock.py`、`graphs/utils/user_profile.py`、
  `agent_test/eval_memory.py` 仍描述旧 store 合并语义
- 改动未提交（等整体改造推进，用户确认后统一提交）
