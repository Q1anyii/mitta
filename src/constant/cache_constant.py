import os

HEALTH_CHECK_INTERVAL = 30
REDIS_INIT_SUCCESS = "Redis服务器初始化成功"
REDIS_CONNECT_FAILED = "Redis服务器连接失败"
REDIS_CONNECT_CLOSED = "Redis服务器连接池已关闭"

TAG_FIELD = "thread_id"
VECTOR_FIELD_NAME = "query_embedding"
VECTOR_FIELD_ALGORITHM = "HNSW"
VECTOR_ATTRIBUTE = {
    "TYPE": "FLOAT32",            # 向量数据类型
    "DIM": 1024,                   # 嵌入向量维度（根据模型而定）
    "DISTANCE_METRIC": "COSINE"   # 距离度量：COSINE / L2 / IP
}

SPARSE_INDEX_NAME = "kb_bm25"
DOC_PREFIX = "kb:doc:"  # 文档 key 前缀

INDEX_NAME = "idx:retrieve_cache"
KEY_PREFIX = "retrieve_cache:"

# 登录态 token 的 Redis key 模板（main.py 签发 / jwt_utils 校验共用，避免两处格式漂移）
USER_TOKEN_KEY = "user:{user_id}:token"
USER_REFRESH_TOKEN_KEY = "user:{user_id}:refresh_token"

CACHE_DEFAULT_TTL = 900

# retrieve_node 节点级缓存 TTL（秒）：LangGraph CachePolicy 声明，缓存后端为 compile(cache=RedisCache)
# 落地的 Redis（见 chat_service.open）；短窗口去重重复检索，可按知识库更新频率调整
CACHE_RETRIEVE_NODE_TTL = 10

# 缓存命中验证阈值：向量初筛召回候选后，用 bge-reranker-v2-m3 验证候选问题与当前问题
# 是否语义等价（实测同义改写 0.89+，无关问题 0.0，取 0.5 有充足间隔）
CACHE_RERANK_HIT_SCORE = 0.5

# ============================================================
# 缓存分层（2026-09-19 H-20260919-10）
# ============================================================
# 背景：原实现只有一层「语义检索缓存」，且 query_cache 每次查询前都要先 embed 一次
#       （没有向量就没法查 KNN），所以命中只能省掉「改写 + 多路召回 + rerank」，
#       embedding 开销一分没少，延迟也只是减半而不是数量级下降。
# 现拆成四层，从上到下代价递增：
#   L1 改写缓存    —— 文本 hash 命中，省一整次 LLM 改写（实测 2694ms，占混合链路 48%）
#   L2 向量缓存    —— 文本 hash 命中，省 embedding API 调用（纯函数，可跨用户共享）
#   L3a 精确结果缓存 —— 文本 hash 命中，省**全部** RAG 链路（0 次 embedding、0 次 rerank）
#   L3b 语义结果缓存 —— LSH 分桶 + KNN 召回 + rerank 验证，省全部 RAG 链路，但需 1 次 embedding
# ============================================================

CACHE_LAYER_ENABLED = os.getenv("MITTA_CACHE_LAYER", "1").strip().lower() in ("1", "true", "on")

# ── L1 查询改写缓存 ──
# key = rw:{prompt_version}:{sha256(归一化(问题 + 历史))}
# 失效：改 REWRITE_PROMPT 时把 REWRITE_PROMPT_VERSION +1，老 key 自然读不到；TTL 兜底 24h。
REWRITE_CACHE_PREFIX = "rw"
REWRITE_PROMPT_VERSION = os.getenv("MITTA_REWRITE_PROMPT_VER", "v1")
REWRITE_CACHE_TTL = int(os.getenv("MITTA_REWRITE_TTL", str(24 * 3600)))

# ── L2 embedding 缓存 ──
# key = emb:{model}:{sha256(归一化(文本))}
# embedding 是纯函数 f(text) -> vector，与用户/会话/线程完全无关，跨用户强共享。
# 失效：换 embedding 模型时 model 名进 key 自动隔离；TTL 7 天 + Redis allkeys-lru 兜底。
EMBED_CACHE_PREFIX = "emb"
EMBED_CACHE_TTL = int(os.getenv("MITTA_EMB_TTL", str(7 * 24 * 3600)))

# ── L3 检索结果缓存 ──
# 知识库版本号：重建/重切 chunk 后 bump，两层结果缓存整体失效（否则会拿旧 chunk 回答）。
KB_VERSION = os.getenv("MITTA_KB_VERSION", "v1")
# L3a 精确层 key = rcache:x:{kb_version}:{sha256(归一化(问题))}（无 thread_id，跨用户共享）
RETRIEVE_EXACT_PREFIX = "rcache:x"
# L3b 语义层沿用原 KEY_PREFIX = "retrieve_cache:"，key 里补 kb_version

# L3b 两阶段验证：先用 KNN top3 跑 rerank，≥ CACHE_RERANK_STRONG_HIT 直接命中（不必把
# top_k=12 全跑一遍）；否则再扩到 top_k 全量验证。用于抵消 top_k 3→12 带来的 rerank 成本。
CACHE_KNN_FAST_K = int(os.getenv("MITTA_CACHE_FAST_K", "3"))
CACHE_RERANK_STRONG_HIT = float(os.getenv("MITTA_CACHE_STRONG_HIT", "0.7"))