"""
Mitta 记忆模块评估脚本（agent_test.eval_memory）
=================================================
评估对象：mem0 检索式长期记忆（P2 写路径 / P3 读路径 / P4 redis 后端 / P5 生命周期加权）。
替代旧版 PostgresStore 全量画像合并评测（该方案已被 P2 移除）。

指标（延迟类均按生产口径统计，warmup 不计入）：
  - 写入延迟：add 事实（mem0 内部 LLM 提取 + embedding 入库）；首次（含建索引/提取）
    与后续分开统计
  - 检索延迟：search（redis 后端自动 BM25+向量混合检索 + P5 加权重排）
  - 检索质量（核心）：add 时带 metadata.category，按 query→预期类别断言 top1/top3 命中率
  - 生命周期：过期记忆（expiration_date）不出现在 search 结果

用法：
    conda activate langchain1.2 / 使用 Python 3.13（需 mem0ai + redisvl）
    cd src
    python -m agent_test.eval_memory                            # 默认 redis 后端
    python -m agent_test.eval_memory --write-n 8                # 减少写入样本（快测）
    python -m agent_test.eval_memory --repeat 3                 # 多轮取中位数
    MEMORY_VECTOR_STORE=chroma python -m agent_test.eval_memory # 本地文件库后端

注意：
  - 本脚本直接操作 MemoryService，不启动 HTTP 服务
  - 测试数据使用独立 user_id（mem_test_*），测试完成后 delete_user 清理
  - 默认 redis 后端（MEMORY_VECTOR_STORE=redis）需要 redis-stack（RediSearch），
    redis 不可用时自动降级（报告将标注 degraded=True）
"""
import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)


# ============================================================
# 统计工具：完整分布 + 多轮中位数
# ============================================================

def _percentile(sorted_lat: List[float], p: float) -> float:
    """线性插值百分位；p=50 -> p50，p=95 -> p95。"""
    if not sorted_lat:
        return 0.0
    if len(sorted_lat) == 1:
        return sorted_lat[0]
    rank = (len(sorted_lat) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(sorted_lat) - 1)
    frac = rank - lo
    return sorted_lat[lo] + (sorted_lat[hi] - sorted_lat[lo]) * frac


def summarize(latencies: List[float]) -> Dict[str, float]:
    """完整分布：avg / p50 / p95 / max / min。"""
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
    s = sorted(latencies)
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p50_ms": _percentile(s, 50),
        "p95_ms": _percentile(s, 95),
        "max_ms": s[-1],
        "min_ms": s[0],
    }


def median_of_runs(runs: List[Dict[str, float]]) -> Dict[str, float]:
    """多轮重复跑：每项指标取中位数（消除单次抖动）。"""
    keys = ("avg_ms", "p50_ms", "p95_ms", "max_ms", "min_ms")
    return {k: sorted(r[k] for r in runs)[len(runs) // 2] for k in keys}


# ============================================================
# 测试数据集：贴近生产的记忆事实（category 用于检索质量断言）
# ============================================================

USERNAME = "评测用户"

# 事实池：(category, 事实原文)。add 时带 metadata={"category": cat}，
# mem0 search 结果保留 metadata，可按类别自动断言命中率。
FACT_POOL = [
    ("职业", "计算机相关专业在读学生，正在寻找 AI 应用开发方向的机会"),
    ("技能", "Java 后端与 Python 双栈，熟悉 Spring Boot、FastAPI"),
    ("技能", "熟悉 LangGraph、LangChain，做过 Agent 状态机与工具编排"),
    ("项目", "自研 AI Agent 项目 Mitta：FastAPI + LangGraph + RAG + MCP"),
    ("知识", "对 RAG 有实践经验：BM25 + 稠密向量 + reranker 混合链路"),
    ("偏好", "偏好深色主题界面，重视代码可读性"),
    ("偏好", "喜欢用 VS Code 作为主力编辑器"),
    ("环境", "使用 conda 管理 Python 环境（langchain1.2）"),
    ("位置", "位于江西南昌"),
    ("目标", "目标岗位是 AI 应用 / Agent 开发相关方向"),
    ("学习", "正在学习多 Agent 协作与 Supervisor 架构"),
    ("兴趣", "对 AI 产品化、Agent 落地场景感兴趣"),
    ("工具", "使用 Docker 与 docker-compose 部署服务"),
    ("工具", "使用 GitHub Actions 做 CI/CD 与部署"),
    ("习惯", "开发中习惯先跑小样本验证再全量执行"),
]

# 检索质量测试：query → 期望命中的类别
RETRIEVAL_QUERIES = [
    ("用户的目标岗位是什么方向？", "目标"),
    ("用户在哪个城市？", "位置"),
    ("用户熟悉哪些技术栈？", "技能"),
    ("用户做过什么项目？", "项目"),
    ("用户对什么感兴趣？", "兴趣"),
    ("用户用什么工具部署？", "工具"),
    ("用户的学习方向？", "学习"),
    ("用户的职业背景？", "职业"),
    ("用户有什么使用偏好？", "偏好"),
    ("用户的开发习惯？", "习惯"),
]


def fact_messages(category: str, fact: str) -> List[Dict[str, str]]:
    """组装 add 消息对（对齐 memory_node._add_memory 的形态）。"""
    return [
        {"role": "user", "content": f"{USERNAME}说：{fact}"},
        {"role": "assistant", "content": f"好的，已记住这条{category}信息。"},
    ]


# ============================================================
# MemoryService 封装（延迟测量 + 数据清理）
# ============================================================

class MemoryServiceTester:
    """MemoryService 测试封装。"""

    def __init__(self):
        from service.memory_service import MemoryService, build_mem0_config

        self._svc_cls = MemoryService
        self._cfg_builder = build_mem0_config
        self.svc: Optional[object] = None
        self.degraded = False

    def open(self):
        try:
            cfg = self._cfg_builder()
            self.svc = self._svc_cls(cfg)
            logger.success(f"MemoryService 就绪（backend={cfg['vector_store']['provider']}）")
        except Exception as e:
            logger.error(f"MemoryService 初始化失败: {e}")
            raise

    def close(self, user_ids: List[str]):
        """清理测试用户记忆（含 P5 访问计数 key 由 _redis_client 维护，一并清理）。"""
        for uid in user_ids:
            try:
                if self.svc:
                    self.svc.delete_user(uid)
            except Exception:
                pass
        # 访问计数 key 清理
        try:
            import redis as redis_py

            url = os.environ.get("REDIS_DB_URL", "redis://localhost:6379/0")
            client = redis_py.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
            for uid in user_ids:
                client.delete(f"mem0:access:{uid}")
        except Exception:
            pass

    def add(self, user_id: str, messages: List[Dict[str, str]], metadata: Optional[Dict] = None) -> Tuple[float, bool]:
        t0 = time.perf_counter()
        ok = self.svc.add(user_id, messages, metadata=metadata)
        return (time.perf_counter() - t0) * 1000, ok

    def search(self, user_id: str, query: str, top_k: int = 5, threshold: Optional[float] = None) -> Tuple[float, List[Dict]]:
        t0 = time.perf_counter()
        results = self.svc.search(query, user_id, top_k=top_k, threshold=threshold)
        return (time.perf_counter() - t0) * 1000, results


# ============================================================
# 延迟测试
# ============================================================

def run_add_latency_once(t: MemoryServiceTester, user_id: str, n: int, warmup: int) -> Dict[str, float]:
    """单轮 add 延迟：首条（建索引/LLM提取）与后续分开统计。"""
    first_ms: Optional[float] = None
    latencies: List[float] = []
    for i in range(warmup + n):
        cat, fact = FACT_POOL[i % len(FACT_POOL)]
        elapsed, ok = t.add(user_id, fact_messages(cat, fact), metadata={"category": cat})
        if not ok:
            t.degraded = True
        if i < warmup:
            continue
        if i == warmup:
            first_ms = elapsed  # 第一条正式样本（可能含建索引开销）
        else:
            latencies.append(elapsed)
    return {
        "first_ms": first_ms or 0.0,
        **summarize(latencies),
    }


def test_add_latency(t: MemoryServiceTester, user_id: str, n: int, warmup: int, repeat: int) -> Dict:
    logger.info(f"【测试 1】add 写入延迟（{n} 次/轮 × {repeat} 轮，warmup {warmup}）")
    runs = [run_add_latency_once(t, user_id, n, warmup) for _ in range(repeat)]
    final = median_of_runs(runs)
    first = sorted(r["first_ms"] for r in runs)[len(runs) // 2]
    result = {
        "agent_test": "add_latency",
        "count_per_run": n,
        "warmup": warmup,
        "repeat": repeat,
        "runs": runs,
        "first_ms": first,
        "avg_ms": final["avg_ms"],
        "p50_ms": final["p50_ms"],
        "p95_ms": final["p95_ms"],
        "max_ms": final["max_ms"],
        "min_ms": final["min_ms"],
    }
    logger.info(
        f"  首条 {first:.0f}ms / 后续 avg={final['avg_ms']:.1f} p95={final['p95_ms']:.1f} max={final['max_ms']:.1f} ms"
    )
    return result


def test_search_latency(t: MemoryServiceTester, user_id: str, n: int, warmup: int, repeat: int) -> Dict:
    logger.info(f"【测试 2】search 检索延迟（{n} 次/轮 × {repeat} 轮，warmup {warmup}）")
    queries = [q for q, _ in RETRIEVAL_QUERIES]

    def once():
        lat = []
        for i in range(warmup + n):
            elapsed, _ = t.search(user_id, queries[i % len(queries)])
            if i >= warmup:
                lat.append(elapsed)
        return summarize(lat)

    runs = [once() for _ in range(repeat)]
    final = median_of_runs(runs)
    result = {
        "agent_test": "search_latency",
        "count_per_run": n,
        "warmup": warmup,
        "repeat": repeat,
        "runs": runs,
        "avg_ms": final["avg_ms"],
        "p50_ms": final["p50_ms"],
        "p95_ms": final["p95_ms"],
        "max_ms": final["max_ms"],
        "min_ms": final["min_ms"],
    }
    logger.info(
        f"  检索分布 avg={final['avg_ms']:.1f} p50={final['p50_ms']:.1f} p95={final['p95_ms']:.1f} ms"
    )
    return result


# ============================================================
# 检索质量（核心）+ 生命周期
# ============================================================

def test_retrieval_quality(t: MemoryServiceTester, user_id: str) -> Dict:
    """【测试 3】检索质量：写入全部事实，按 query→预期类别断言命中率。

    口径对齐生产（llm_node 注入：top_k=5, threshold=0.3）——top5 命中率即真实召回覆盖。
    """
    logger.info("【测试 3】检索质量（top1/top3/top5 命中率，threshold=0.3 对齐生产）")
    for cat, fact in FACT_POOL:
        t.add(user_id, fact_messages(cat, fact), metadata={"category": cat})

    cases = []
    top1_hits = top3_hits = top5_hits = total = 0
    for query, expect_cat in RETRIEVAL_QUERIES:
        _, results = t.search(user_id, query, top_k=5, threshold=0.3)
        cats = [r.get("metadata", {}).get("category", "") if isinstance(r.get("metadata"), dict) else ""
                for r in results]
        top1_hit = bool(cats) and cats[0] == expect_cat
        top3_hit = expect_cat in cats[:3]
        top5_hit = expect_cat in cats[:5]
        top1_hits += int(top1_hit)
        top3_hits += int(top3_hit)
        top5_hits += int(top5_hit)
        total += 1
        cases.append({
            "query": query,
            "expect_category": expect_cat,
            "top5_categories": cats[:5],
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
        })
        logger.info(f"  q={query} expect={expect_cat} → top5={cats[:5]} {'✓' if top5_hit else '✗'}")

    result = {
        "agent_test": "retrieval_quality",
        "threshold": 0.3,
        "cases": total,
        "top1_hit_rate": round(top1_hits / total, 3),
        "top3_hit_rate": round(top3_hits / total, 3),
        "top5_hit_rate": round(top5_hits / total, 3),
        "details": cases,
    }
    logger.info(f"  top1 命中率 {top1_hits}/{total} = {result['top1_hit_rate']*100:.0f}%")
    logger.info(f"  top3 命中率 {top3_hits}/{total} = {result['top3_hit_rate']*100:.0f}%")
    logger.info(f"  top5 命中率 {top5_hits}/{total} = {result['top5_hit_rate']*100:.0f}%")
    return result


def test_expiration(t: MemoryServiceTester, user_id: str) -> Dict:
    """【测试 4】生命周期：已过期记忆不出现在检索结果，未过期可见。"""
    logger.info("【测试 4】过期清理（expiration_date）")
    from datetime import date, timedelta

    # 已过期（昨天过期）——直接底层调用写过去日期
    t.svc.memory.add(
        fact_messages("偏好", "评测临时偏好：喜欢喝美式咖啡"),
        user_id=user_id,
        metadata={"category": "偏好"},
        expiration_date=(date.today() - timedelta(days=1)).isoformat(),
    )
    # 未过期（显式关闭过期）
    t.svc.add(
        user_id,
        fact_messages("偏好", "长期偏好：喜欢喝拿铁咖啡"),
        metadata={"category": "偏好"},
        expiration_days=0,
    )
    _, results = t.search(user_id, "用户喜欢喝什么咖啡？", top_k=5)
    hit_expired = any("美式咖啡" in r.get("text", "") for r in results)
    hit_keep = any("拿铁咖啡" in r.get("text", "") for r in results)
    result = {
        "agent_test": "expiration",
        "expired_hidden": not hit_expired,
        "keep_visible": hit_keep,
        "top_results": [r.get("text", "")[:40] for r in results[:3]],
    }
    logger.info(f"  已过期记忆被隐藏: {result['expired_hidden']} / 长期记忆可见: {result['keep_visible']}")
    return result


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mitta 记忆模块评估（mem0 检索式）")
    parser.add_argument("--user-id", type=str, default=f"mem_test_{uuid.uuid4().hex[:8]}", help="测试用户 ID")
    parser.add_argument("--include-latency", action="store_true", help="额外跑写入/检索延迟（默认只跑质量+生命周期）")
    parser.add_argument("--write-n", type=int, default=8, help="add 样本量/轮（--include-latency 时生效，默认 8）")
    parser.add_argument("--read-n", type=int, default=30, help="search 样本量/轮（--include-latency 时生效，默认 30）")
    parser.add_argument("--warmup", type=int, default=1, help="warmup 次数（--include-latency 时生效，默认 1）")
    parser.add_argument("--repeat", type=int, default=2, help="重复轮数取中位数（--include-latency 时生效，默认 2）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Mitta 记忆模块评估（mem0 检索式，P2/P3/P4/P5）")
    logger.info(f"用户 ID: {args.user_id} | 后端: {os.environ.get('MEMORY_VECTOR_STORE', 'chroma(默认)')}")
    logger.info("=" * 60)

    t = MemoryServiceTester()
    try:
        t.open()
    except Exception as e:
        logger.error(f"MemoryService 初始化失败: {e}")
        return

    all_results = []
    try:
        if args.include_latency:
            all_results.append(test_add_latency(t, args.user_id, args.write_n, args.warmup, args.repeat))
            all_results.append(test_search_latency(t, args.user_id, args.read_n, args.warmup, args.repeat))
        all_results.append(test_retrieval_quality(t, args.user_id))
        all_results.append(test_expiration(t, args.user_id))
    finally:
        t.close([args.user_id])
        logger.info("测试数据已清理")

    all_results.append({"agent_test": "meta", "degraded": t.degraded})

    # ---- 汇总报告 ----
    logger.info(f"\n{'='*60}")
    logger.info("【记忆模块评估汇总】")
    for r in all_results:
        name = r.get("agent_test", "unknown")
        if name == "add_latency":
            logger.info(f"  add 写入: 首条 {r['first_ms']:.0f}ms / 后续 p50 {r['p50_ms']:.1f}ms p95 {r['p95_ms']:.1f}ms")
        elif name == "search_latency":
            logger.info(f"  search 检索: p50 {r['p50_ms']:.1f}ms p95 {r['p95_ms']:.1f}ms")
        elif name == "retrieval_quality":
            logger.info(f"  检索质量: top1 {r['top1_hit_rate']*100:.0f}% / top3 {r['top3_hit_rate']*100:.0f}% / top5 {r['top5_hit_rate']*100:.0f}%")
        elif name == "expiration":
            logger.info(f"  生命周期: 过期隐藏={r['expired_hidden']} 长期可见={r['keep_visible']}")
    if t.degraded:
        logger.warning("注意：部分操作降级（后端不可用），指标可能不代表生产真实值")

    output_path = Path(__file__).parent / "memory_eval_report.json"
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"\n评估报告已保存: {output_path}")


if __name__ == "__main__":
    main()
