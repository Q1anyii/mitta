"""Mem0 记忆服务适配层（改造路线 Phase 1 增量）。

把 mem0 接入 Mitta 的最小零侵入入口：只提供构造与包装，不接入 main_graph。
写/读路径的接入见改造路线 Phase 2/3（memory_node 替换、llm_node 注入）。

设计要点：
- mem0 延迟导入 + 缺失降级：生产环境（conda/Docker）未装 mem0 时，
  import 本模块不报错，调用返回空结果，保证现有主链路不受影响。
- 后端可切：MEMORY_VECTOR_STORE=chroma（默认本地文件库）| redis（RedisSearch，
  需 redis-stack 且 pip install redisvl）。redis 建议独立 db（如 db=1），
  避免与业务缓存/限流共享 LRU 驱逐导致记忆被逐出（不可重建）。
- 单例懒加载：首次调用才构建（chroma PersistentClient 构建较重），失败静默降级。

与 featrue_demo/memory_config.py 保持同一模型选型：DeepSeek chat LLM
+ SiliconFlow bge-m3 embedding（OpenAI 兼容接口）。
"""

import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# 生产环境 .env 由 config.py 统一 load_dotenv(override=True)；
# 独立脚本/评测脚本运行时防御性加载项目根 .env（幂等）。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_gate = threading.Lock()
_singleton: Optional["MemoryService"] = None

# ---- P5 生命周期/时间加权参数（可被环境变量覆盖）----
# 默认记忆有效期（天）：过期后 search/get_all 自动隐藏（mem0 原生）。
_DEFAULT_TTL_DAYS = 90
# 时间衰减窗口（天）：created_at 距今越近，额外加分越高（0→满窗线性）。
_RECENCY_WINDOW_DAYS = 30
# 访问加权（LRU 语义）：距上次被检索越近，额外加分越高；窗口（天）。
_ACCESS_WINDOW_DAYS = 7
# 加分权重（相对 mem0 原始 score 0~1 的量级，不反转相关性排序）。
_W_RECENCY = 0.15
_W_ACCESS = 0.25
# 访问计数在业务 redis 的 key 与保留期（丢了只损失加权，可重建，无碍）。
_ACCESS_KEY_TPL = "mem0:access:{}"
_ACCESS_TTL_SECONDS = _ACCESS_WINDOW_DAYS * 86400


def _patch_mem0_redis_compat() -> None:
    """桥接 mem0 2.2.1 与 redisvl 0.28.0 的 API 差异（仅 redis 后端需要）。

    mem0 的 RedisDB 用旧 redisvl API：SearchIndex.from_dict(schema) + set_client(client)；
    redisvl>=0.3 移除 set_client（改用构造参数 redis_client）。项目保持
    redis==8.1.0 + redisvl 0.28.0（降 redis 会影响业务缓存/BM25/限流），
    此处运行时补丁桥接：set_client 等价于在构造时注入 redis_client。
    """
    try:
        from redisvl.index.index import SearchIndex
    except ImportError:
        return
    if hasattr(SearchIndex, "set_client"):
        return

    def _set_client(self, client) -> None:
        # name-mangled 私有属性（_SearchIndex__redis_client）为 redisvl 0.28.0
        # 内部连接存储；_client_was_injected 标记外部注入、_owns_redis_client 关闭
        # 自动关闭语义，与 __init__(redis_client=...) 行为一致。
        object.__setattr__(self, "_SearchIndex__redis_client", client)
        self._client_was_injected = True
        self._owns_redis_client = False

    SearchIndex.set_client = _set_client
    logger.info("mem0 redis 兼容补丁已应用（SearchIndex.set_client）")


def build_mem0_config() -> dict:
    """构造 mem0 配置（对齐 memory_config.py 的模型选型）。

    后端由环境变量 MEMORY_VECTOR_STORE 决定：chroma（默认）| redis。
    redis 时 embedding_model_dims 必须 = 1024（bge-m3 维度），否则索引不匹配。
    """
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("SILICONFLOW_API_KEY"):
        dotenv_path = _PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path, override=True)
            except ImportError:
                logger.warning("dotenv 未安装，跳过 .env 加载（生产环境由 config.py 负责）")

    backend = os.environ.get("MEMORY_VECTOR_STORE", "chroma").strip().lower()

    if backend == "redis":
        _patch_mem0_redis_compat()
        vector_store = {
            "provider": "redis",
            "config": {
                # RediSearch FT 索引只能在 db 0 创建（实测：Cannot create index on db != 0），
                # 隔离不能靠独立 db，必须独立实例（compose redis-memory，noeviction 防驱逐，
                # 避免业务 redis 的 allkeys-lru 驱逐记忆；默认 db 0）。
                "redis_url": os.environ.get(
                    "MEMORY_REDIS_URL", os.environ.get("REDIS_DB_URL", "redis://localhost:6379/0")
                ),
                "collection_name": os.environ.get("MEMORY_COLLECTION", "mem0"),
                "embedding_model_dims": 1024,  # bge-m3 维度
            },
        }
        # 注：mem0 2.2.1 的 redis 后端，search 内部自动执行 BM25（keyword_search）
        # + 向量混合检索，无需额外代码（见 mem0/memory/main.py _search_vector_store）
    else:
        vector_store = {
            "provider": "chroma",
            "config": {
                "collection_name": os.environ.get("MEMORY_COLLECTION", "mem0"),
                "path": str(_PROJECT_ROOT / "resources" / "mem0_chroma"),
            },
        }

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "deepseek-chat",
                "api_key": os.environ["DEEPSEEK_API_KEY"],
                "openai_base_url": "https://api.deepseek.com",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "BAAI/bge-m3",
                "api_key": os.environ["SILICONFLOW_API_KEY"],
                "openai_base_url": os.environ.get(
                    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
                ),
            },
        },
        "vector_store": vector_store,
    }


class MemoryService:
    """mem0 长期记忆的薄包装：add / search / get_all / delete_user / reset。

    所有方法失败时降级返回空结果并记日志，不向主链路抛异常
    （延续 cache_service「记忆/缓存是优化而非硬依赖」的项目哲学）。
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or build_mem0_config()
        self._memory = None

    @property
    def memory(self):
        """懒加载 mem0.Memory；mem0 未安装或初始化失败时返回 None（调用方自行降级）。"""
        if self._memory is None:
            try:
                from mem0 import Memory

                self._memory = Memory.from_config(self._config)
            except ImportError:
                logger.warning("mem0 未安装，记忆服务不可用（pip install mem0ai 后启用）")
            except Exception as e:
                logger.error(f"mem0 初始化失败（降级为无记忆）: {e}")
        return self._memory

    def add(
        self,
        user_id: str,
        messages: list[dict],
        metadata: Optional[dict] = None,
        expiration_days: Optional[int] = None,
    ) -> bool:
        """写入记忆：mem0 内部完成 LLM 事实提取 + embedding 入库（异步无回调）。

        生命周期：默认 MEMORY_TTL_DAYS（未设=90）天后过期（mem0 原生 expiration_date，
        search/get_all 自动隐藏过期记忆）；传 expiration_days=0 或负数可显式关闭过期。

        messages: [{"role": "user"/"assistant", "content": ...}, ...]
        返回 True 表示已提交调用（mem0 内部成功与否无法同步确认，失败记日志）。
        """
        m = self.memory
        if m is None:
            return False
        try:
            if expiration_days is None:
                try:
                    expiration_days = int(os.environ.get("MEMORY_TTL_DAYS", _DEFAULT_TTL_DAYS))
                except ValueError:
                    expiration_days = _DEFAULT_TTL_DAYS
            expiration_date = None
            if expiration_days is not None and expiration_days > 0:
                expiration_date = (date.today() + timedelta(days=expiration_days)).isoformat()
            m.add(messages, user_id=user_id, metadata=metadata, expiration_date=expiration_date)
            return True
        except Exception as e:
            logger.error(f"mem0.add 失败 user_id={user_id}: {e}")
            return False

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        reference_date: Optional[str] = None,
    ) -> list[dict]:
        """检索记忆（按 user_id 隔离）。返回 [{"id","text","score","created_at"}]，
        按加权分降序（score = mem0 原始分 + 时间衰减加分 + 访问加权加分）。

        redis 后端时 mem0 search 自动做 BM25+向量混合检索。
        时间衰减：created_at 距今 ≤30 天线性加分（越新越高）；
        访问加权（LRU 语义）：被检索命中的记忆记录 last_access（业务 redis），
        距上次访问越近加分越高（≤7 天窗口）——redis 不可用时自动降级为仅时间衰减。
        reference_date（平台时间衰减）OSS 版不可用，传参会由 mem0 抛错并降级返回 []。
        """
        m = self.memory
        if m is None:
            return []
        try:
            results = m.search(
                query,
                filters={"user_id": user_id},
                top_k=top_k,
                threshold=threshold or 0.1,
                reference_date=reference_date,
            )
            items = [
                {
                    "id": r.get("id"),
                    "text": r.get("memory", ""),
                    "score": r.get("score", 0.0),
                    "created_at": r.get("created_at"),
                }
                for r in results.get("results", [])
            ]
            return self._reweight(items, user_id)
        except Exception as e:
            logger.error(f"mem0.search 失败 user_id={user_id}: {e}")
            return []

    def _redis_client(self):
        """访问计数用业务 redis（REDIS_DB_URL）。不可用时返回 None（降级无访问加权）。

        懒探测一次并缓存结果（None/False 表示不可用，避免每轮重连）。
        """
        if not hasattr(self, "_access_redis"):
            self._access_redis = None
            url = os.environ.get("REDIS_DB_URL", "redis://localhost:6379/0")
            try:
                import redis as redis_py

                client = redis_py.Redis.from_url(
                    url, socket_timeout=2, socket_connect_timeout=2, decode_responses=True
                )
                client.ping()
                self._access_redis = client
            except Exception as e:
                logger.debug(f"记忆访问计数 redis 不可用（降级为仅时间衰减）: {e}")
                self._access_redis = False
        return self._access_redis if self._access_redis else None

    def _reweight(self, items: list[dict], user_id: str) -> list[dict]:
        """P5 手动时间加权：score = 原始分 + 时间衰减加分 + 访问加权（LRU）加分。

        返回按加权分降序的新列表（score 字段为加权后分值，原分保留在 raw_score）。
        """
        if not items:
            return []
        now = time.time()
        rc = self._redis_client()

        # 读取本次命中记忆的 last_access
        access_map: dict = {}
        if rc:
            try:
                vals = rc.hmget(_ACCESS_KEY_TPL.format(user_id), [it.get("id") for it in items])
                access_map = {
                    it.get("id"): float(v)
                    for it, v in zip(items, vals)
                    if v and it.get("id")
                }
            except Exception:
                access_map = {}

        weighted = []
        for it in items:
            bonus = 0.0
            ca = it.get("created_at")
            if ca:
                try:
                    created = datetime.fromisoformat(ca)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
                    if 0 <= age_days <= _RECENCY_WINDOW_DAYS:
                        bonus += _W_RECENCY * (1 - age_days / _RECENCY_WINDOW_DAYS)
                except (ValueError, TypeError):
                    pass
            if access_map and it.get("id") in access_map:
                ago_days = (now - access_map[it["id"]]) / 86400
                if 0 <= ago_days <= _ACCESS_WINDOW_DAYS:
                    bonus += _W_ACCESS * (1 - ago_days / _ACCESS_WINDOW_DAYS)
            weighted.append({**it, "raw_score": it["score"], "score": round(it["score"] + bonus, 4)})

        weighted.sort(key=lambda x: x["score"], reverse=True)

        # fire-and-forget 记录本次命中记忆的访问时间（失败静默，丢失只损失加权）
        if rc and items:
            try:
                pipe = rc.pipeline()
                key = _ACCESS_KEY_TPL.format(user_id)
                for it in items:
                    if it.get("id"):
                        pipe.hset(key, it["id"], now)
                pipe.expire(key, _ACCESS_TTL_SECONDS)
                pipe.execute()
            except Exception:
                pass
        return weighted

    def get_all(self, user_id: str, top_k: int = 100) -> list[dict]:
        """列出该用户全部记忆（评测/排查用）。"""
        m = self.memory
        if m is None:
            return []
        try:
            results = m.get_all(filters={"user_id": user_id}, top_k=top_k)
            return [
                {"text": r.get("memory", ""), "score": r.get("score", 0.0)}
                for r in results.get("results", [])
            ]
        except Exception as e:
            logger.error(f"mem0.get_all 失败 user_id={user_id}: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        """删除该用户全部记忆（数据清理/评测用）。"""
        m = self.memory
        if m is None:
            return False
        try:
            m.delete_all(user_id=user_id)
            return True
        except Exception as e:
            logger.error(f"mem0.delete_all 失败 user_id={user_id}: {e}")
            return False

    def reset(self) -> bool:
        """清空整个记忆库（collection）。"""
        m = self.memory
        if m is None:
            return False
        try:
            m.reset()
            return True
        except Exception as e:
            logger.error(f"mem0.reset 失败: {e}")
            return False


def get_memory_service() -> Optional[MemoryService]:
    """进程内单例；初始化失败返回 None（主链路降级用）。"""
    global _singleton
    if _singleton is None:
        with _gate:
            if _singleton is None:
                try:
                    _singleton = MemoryService()
                except Exception as e:
                    logger.error(f"记忆服务初始化失败（返回 None 降级）: {e}")
                    _singleton = None
    return _singleton
