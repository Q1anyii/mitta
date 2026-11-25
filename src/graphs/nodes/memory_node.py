"""记忆节点：把本轮对话提交 mem0 提取长期事实并入库（按用户隔离）。

P2 改造：原「读全量档案 → LLM 合并 → store.put」整体替换为 mem0.add
（mem0 内部完成事实提取 + embedding 入库，增量而非全量合并）。
store 的 user_profile 稳定档案由 llm_node._ensure_username_profile 独占维护，
本节点不再经手 store。节点无 state 写入，不做节点级缓存；重复事实由 mem0
的 hash 去重兜底。
依赖：memory_service（Mem0 适配层，依赖注入）。
"""

from langchain_core.runnables import RunnableConfig
from loguru import logger
from concurrent.futures.thread import ThreadPoolExecutor

from graphs.state import OverAllState

# 实例化线程池，限制最大并发8个后台任务，防止线程爆炸
executor = ThreadPoolExecutor(max_workers=8)


def memory_node(state: OverAllState, config: RunnableConfig, memory_service) -> None:
    """将本轮对话中的长期信息提取并写入 mem0（按用户隔离）。

    快速路径：idle 轮（筛选出工具但模型未调用，通常是简单闲聊）且无需检索时，
    直接跳过，避免 mem0 提取调用阻塞 SSE 流。用户名基础档案已由 llm_node 在回答前写入，
    闲聊场景通常无新增长期信息。
    fire-and-forget：真正需要提取的轮次，把 mem0.add（内部 LLM 事实提取 +
    embedding 入库）放入后台线程，节点函数立即返回，不再阻塞 SSE 完成态。
    短期记忆入库由 checkpointer 在图机制内完成。

    Args:
        state: 当前图状态
        config: LangGraph 配置（含 user_id）
        memory_service: Mem0 记忆服务（依赖注入，见 service.memory_service）
    """
    user_id = config["configurable"].get("user_id", "default")
    if not state["messages"]:
        return
    if state.get("tool_status") in ("idle", "unavailable") and not state.get("needs_retrieval"):
        logger.debug(f"memory_node 跳过（{state.get('tool_status')} 轮，无新增长期信息）user_id={user_id}")
        return
    input_str = state["input_str"]
    ai_reply = state["messages"][-1].content

    def _add_memory(uid: str, in_str: str, llm_out: str):
        # mem0.add 输入 = 本轮对话；内部 LLM 事实提取 + embedding 入库，失败静默降级
        messages = [
            {"role": "user", "content": in_str},
            {"role": "assistant", "content": llm_out},
        ]
        memory_service.add(uid, messages)

    executor.submit(_add_memory, user_id, input_str, ai_reply)
    return
