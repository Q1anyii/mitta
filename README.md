# Mitta AI 智能助理（米塔）

面向自定义知识库的 AI 智能助理：基于 **LangGraph + RAG + MCP + 流式 SSE**，内置完整检索链路（查询改写 → 多路召回 → RRF 融合 → 在线重排），支持三层记忆（短期会话 checkpoint / 用户名低频档案 / mem0 检索式长期事实，叠加+乘性时间衰减+90 天 TTL）、MCP 工具调用（文件系统/数据库/网页抓取等）、知识库增量入库，并通过 SSE 流式输出实现打字机效果。已上线公网（HTTPS）并提供 Tauri 桌面端。

## UI 演示

![Mitta 登录演示](docs/assets/登录演示.gif)

![Mitta 聊天演示](docs/assets/聊天演示.gif)

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Q1anyii/Mitta.git Mitta && cd Mitta

# 2. 配置环境变量（DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / POSTGRESQL_DB_URL / REDIS_DB_URL / JWT_SECRET_KEY）
cp .env.example .env

# 3. 一键启动（PostgreSQL + Redis + API + Nginx；低配服务器改用 ChromaDB 免 Milvus）
docker-compose up -d
```

访问 `http://localhost`。开发模式：`cd src && uvicorn main:app --port 8000 --reload`（后端托管 SPA）。详细步骤（MCP 配置 / 知识库入库 / 向量库切换 / 部署运维）见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

## 系统架构

### 整体架构

<p align="center">
  <img src="docs/figures/architecture.svg" alt="Mitta 整体架构图" width="95%">
</p>

### 主对话图（main_graph）

<p align="center">
  <img src="docs/figures/main-graph.svg" alt="Mitta 主对话图" width="95%">
</p>

### RAG 检索子图（retrieve_graph）

<p align="center">
  <img src="docs/figures/retrieve-graph.svg" alt="Mitta RAG 检索子图" width="95%">
</p>

### CI/CD 流水线

<p align="center">
  <img src="docs/figures/ci-flow.svg" alt="Mitta CI/CD 流水线" width="95%">
</p>

> 各图节点说明、条件路由、并行编排、关键参数与设计思路见 [docs/architecture-flowcharts.md](docs/architecture-flowcharts.md)。

## 技术栈

| 层次 | 技术 |
|---|---|
| 语言/环境 | Python 3.12 |
| Agent 编排 | LangGraph 1.x（StateGraph / Send 条件路由 / CachePolicy / Checkpointer / Store） |
| LLM | LangChain 1.x + DeepSeek（deepseek-flash = V4.1-Flash，OpenAI 兼容，支持 reasoning_content） |
| Embedding / 重排 | SiliconFlow `BAAI/bge-m3`（1024 维）/ `BAAI/bge-reranker-v2-m3` |
| 向量库 | ChromaDB（默认，免部署）/ Milvus（可插拔） |
| 存储 | PostgreSQL 16（会话 checkpoint + 用户表）、Redis 7（缓存 + RedisSearch BM25 + 限流 + 登录态 + mem0 记忆库可选后端）、ChromaDB（mem0 默认本地向量库） |
| MCP | MCP Python SDK + FastMCP（内置 agent_server + 外部 stdio/sse 服务器） |
| Web | FastAPI + Uvicorn（SSE 流式）+ Nginx（静态托管 + API 代理） |
| 前端 | Vue 3 运行时（CDN，无构建工具 / 无 SFC）+ 手写设计系统 + 多主题 + 响应式 |
| 认证 | JWT（双 token 无感续签）+ bcrypt |

## 功能特性

统一意图路由 · 多人格路由（4 人格 + 工具白名单 + chibi 吐槽）· RAG 混合检索 · MCP 工具集成（分组 + 懒加载）· 智能工具筛选 · 三层记忆（会话 checkpoint / PG 用户档案 / mem0 检索式长期事实）· 检索缓存四层（LSH + KNN + rerank，TTL 抖动 + singleflight 防雪崩）· 流式 SSE + ack 预响应 · 断点续传 · 文件上传解析 · 知识库增量更新 API · 用户级 MCP 热重载 · 深度思考 · 安全认证。完整逐条说明见 [docs/FEATURES.md](docs/FEATURES.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/FEATURES.md](docs/FEATURES.md) | 功能特性逐条、工程亮点、评测指标速览 |
| [docs/AGENT_EVAL_MATRIX.md](docs/AGENT_EVAL_MATRIX.md) | Agent 系统评测矩阵 E1–E15（路由/工具/缓存/检索/记忆/限流/认证/SSE/在线实测）与全部实测记录 |
| [docs/architecture-flowcharts.md](docs/architecture-flowcharts.md) | 主图/检索子图节点说明、条件路由、并行编排、关键参数、混合检索设计思路 |
| [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) | 核心设计说明（MCP 常驻事件循环/缓存四层/记忆异步化/安全设计/热重载/深度思考） |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 快速开始完整步骤、Docker 部署、CI/CD 流水线与蓝绿入库 |
| [docs/API.md](docs/API.md) | API 接口一览（认证/对话/用户/MCP/知识库） |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | 目录结构 |
| [docs/BACKLOG.md](docs/BACKLOG.md) | 后续规划 |
| [docs/devlog/](docs/devlog/) | 开发日志 |

## 贡献与许可

欢迎提交 Issue 和 Pull Request，见 [CONTRIBUTING.md](CONTRIBUTING.md)。许可证见 [LICENSE](LICENSE)。