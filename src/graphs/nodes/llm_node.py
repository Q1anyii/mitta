"""核心生成节点：组装提示词 → 工具筛选 → LLM 流式生成 → 合并 chunk。

工具循环防护（次数上限/重复调用/失败熔断）已抽到 llm_circuit.py，
本节点只负责编排。
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from loguru import logger

from graphs.state import OverAllState
from graphs.tool_filter import ToolFilter
from constant.persona_constant import DEFAULT_PERSONA, get_persona
from graphs.utils.history_repair import _repair_history, _trim_history
from graphs.utils.user_profile import _ensure_username_profile, _get_username
from graphs.utils.llm_circuit import (
    MAX_RETRIEVAL_DOCS,
    MAX_DOC_CHARS,
    compute_circuit,
)


def llm_node(
    state: OverAllState,
    config: RunnableConfig,
    store: BaseStore,
    model,
    system_prompt: str,
    tool_filter: ToolFilter,
    tools: list[BaseTool],
    get_user_system_prompt,
    lazy_loader=None,
    memory_service = None
) -> OverAllState:
    """核心生成节点：组装上下文 → 筛选工具 → LLM 生成 → 返回消息。"""
    logger.success("llm_node is runed")
    input_str = state["input_str"]
    retrieval_res = state.get("retrieve_res")

    # ── 1. 组装用户消息（检索分支 / 无需检索分支）──
    if retrieval_res and "output" in retrieval_res:
        raw_docs = retrieval_res.get("output", [])
        docs = [
            doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
            for doc in raw_docs
        ]
        if docs:
            docs = [d[:MAX_DOC_CHARS] for d in docs[:MAX_RETRIEVAL_DOCS]]
            context = "\n\n".join(f"[文档 {i + 1}] {doc}" for i, doc in enumerate(docs))
        else:
            context = "（知识库中未检索到相关内容）"
        user_content = (
            f"请严格依据下面检索到的资料回答用户问题，资料中没有的内容不要编造。\n\n"
            f"【引用要求】\n"
            f"- 事实性陈述（数字、接口、配置、机制、结论）后必须标注来源角标，如 [1][2]，"
            f"角标编号对应下方【检索资料】的 [文档 i]；\n"
            f"- 若所有资料均无法支撑某个事实，明确说\"知识库暂未覆盖该内容\"，不要用常识补全；\n"
            f"- 角标只标事实句，客套话/过渡句/总结句不加。\n\n"
            f"【检索资料】\n{context}\n\n"
            f"【用户问题】\n{input_str}"
        )
    else:
        user_content = input_str

    # ── 2. 长期记忆 ──
    user_id = config["configurable"].get("user_id", "default")
    username = _get_username(config)
    long_term = _ensure_username_profile(store, user_id, username)
    # P3：mem0 检索式记忆（按需注入，mem0 未装/失败降级为空）
    retrieved_memory = []
    if memory_service is not None:
        try:
            retrieved_memory = memory_service.search(
                query=input_str,
                user_id=user_id,
                top_k=5,
                threshold=0.5,
            ) or []
        except Exception as e:
            logger.warning(f"mem0 记忆检索降级（user_id={user_id}）：{e}")
            retrieved_memory = []

    # ── 3. 短期记忆 ──
    history = _trim_history(list(state.get("messages", [])))

    # ── 4. 组装 system prompt ──
    system_content = get_user_system_prompt(user_id, system_prompt)
    persona_key = state.get("persona") or DEFAULT_PERSONA
    persona = get_persona(persona_key)
    is_default_persona = persona_key == DEFAULT_PERSONA
    system_content += f"\n\n{persona['prompt']}"
    if long_term and long_term != "（暂无档案）":
        system_content += f"\n\n【用户长期记忆】\n{long_term}"
    if retrieved_memory:
        system_content += "\n\n【用户近期记忆】\n" + "\n".join(
            f"- {m['text']}" for m in retrieved_memory
        )

    messages = [SystemMessage(content=system_content)] + _repair_history(list(history))
    messages.append(HumanMessage(content=user_content))

    # ── 5. 运行时工具筛选 ──
    last_ai = next(
        (m.content for m in reversed(history) if isinstance(m, AIMessage) and m.content),
        "",
    )
    if last_ai and re.search(r"(继续|刚才|那个|这个|它|同样|跟刚才)", input_str):
        filter_query = f"{input_str}\n{last_ai[:200]}"
    else:
        filter_query = input_str

    selected_tools = tool_filter.select_tools(filter_query, tools)

    # 懒加载触发：语义命中未连接的第三方 MCP 工具 → 拉起连接 → 重试筛选
    pending_hits = list(getattr(tool_filter, "last_pending_hits", []) or [])
    if pending_hits and lazy_loader is not None:
        server_name = lazy_loader.server_of(pending_hits[0])
        if server_name:
            logger.info(f"工具筛选命中懒加载 server [{server_name}]（{pending_hits}），触发连接")
            if lazy_loader.trigger(server_name):
                selected_tools = tool_filter.select_tools(filter_query, tools)

    # 人格白名单
    allowed = persona["allowed_tools"]
    if allowed is not None:
        allowed_set = set(allowed)
        selected_tools = [t for t in selected_tools if t.name in allowed_set]
    logger.info(
        f"[persona] key={persona_key} label={persona['label']} "
        f"tools_in={len(tools)} tools_filtered={len(selected_tools)} "
        f"white_list={'none(全量)' if allowed is None else f'{len(allowed)}个'}"
    )

    # ── 5.2 工具循环防护（委托 llm_circuit：次数上限/重复调用/失败熔断）──
    circuit = compute_circuit(history, selected_tools)
    force_stop = circuit["force_stop"]
    broken_tools = circuit["broken_tools"]

    if broken_tools:
        logger.warning(f"工具失败熔断：{sorted(broken_tools)} 本轮已禁用")

    if selected_tools and not force_stop:
        model_with_tools = model.bind_tools(selected_tools)
    else:
        if force_stop:
            logger.warning(f"本轮强制停止工具调用：{circuit['stop_reason']}")
            messages.append(SystemMessage(
                content="注意：本轮请求中工具调用次数已达上限，请不要再调用任何工具，"
                        "直接基于你已有的上下文信息回答用户的问题。"
            ))
        elif broken_tools:
            messages.append(SystemMessage(
                content=f"注意：以下工具本轮已连续失败多次、暂时不可用：{'、'.join(sorted(broken_tools))}。"
                        "请不要再尝试调用它们，改用其他方式回答用户，或如实说明这部分暂时做不到。"
            ))
        elif not is_default_persona and allowed == []:
            logger.info(f"[persona] {persona_key} 人格主动无工具，裸模型对话")
        else:
            messages.append(SystemMessage(
                content="注意：当前没有可用的工具。若用户的请求依赖工具能力（如查文件、查数据库、"
                        "操作外部服务），请如实告知暂时无法处理，不要编造结果或假装已执行。"
            ))
        model_with_tools = model

    # ── 5.5 深度思考模式 ──
    thinking_mode = config["configurable"].get("thinking_mode", False)
    reasoning_effort = config["configurable"].get("reasoning_effort", "low")
    logger.info(f"[thinking-debug] thinking_mode={thinking_mode} reasoning_effort={reasoning_effort}")
    if thinking_mode:
        model_with_tools = model_with_tools.bind(
            extra_body={"thinking": {"type": "enabled"}},
            reasoning_effort=reasoning_effort,
        )
    else:
        # DeepSeek V4.1-Flash 默认思考开关为 enabled；不传 disabled 会白烧推理 token
        # 且使 temperature 等参数失效，故非思考模式必须显式 bind disabled
        model_with_tools = model_with_tools.bind(
            extra_body={"thinking": {"type": "disabled"}},
        )

    # ── 6. 流式生成 ──
    chunks = []
    for chunk in model_with_tools.stream(messages):
        chunks.append(chunk)

    final_chunk = None
    for c in chunks:
        final_chunk = c if final_chunk is None else final_chunk + c
    content = final_chunk.content if final_chunk else ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    tool_calls = final_chunk.tool_calls if final_chunk else []

    # 熔断硬约束：次数上限/重复调用时代码层直接剥空 tool_calls，不靠提示词
    if force_stop and tool_calls:
        logger.warning(f"熔断硬剥离：模型仍试图调用工具 {[tc['name'] for tc in tool_calls]}，已强制清空")
        tool_calls = []

    logger.info(
        f"llm_node 生成完成：tool_calls={[tc['name'] for tc in tool_calls]} "
        f"content_len={len(content)}"
    )
    ai_reply = AIMessage(content=content, tool_calls=tool_calls)

    # ── 7. 本轮工具状态 ──
    tool_status = "idle"
    if tool_calls:
        tool_status = "executed"
    elif history and isinstance(history[-1], ToolMessage):
        tool_status = "executed"
    elif not selected_tools:
        tool_status = "unavailable"

    # ── 8. 工具循环轮次去重用户消息 ──
    in_tool_loop = bool(history) and isinstance(history[-1], ToolMessage)
    if in_tool_loop:
        return {"messages": [ai_reply], "tool_status": tool_status}
    return {"messages": [HumanMessage(content=input_str), ai_reply], "tool_status": tool_status}
