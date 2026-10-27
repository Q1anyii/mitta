import os
from pathlib import Path
from dotenv import load_dotenv


# 加载项目根 .env（DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL）
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# mem0 默认配置的 LLM 与 Embedder 都是 OpenAI 官方（必须 OPENAI_API_KEY）。
# Mitta 实际用的是 DeepSeek（LLM）+ SiliconFlow（bge-m3 embedding，OpenAI 兼容接口），
# 这里显式注入，与项目 container.py 的模型选型保持一致。
config = {
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
            "openai_base_url": os.environ["SILICONFLOW_BASE_URL"],
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_demo",
            "path": str(Path(__file__).resolve().parents[2] / "resources" / "mem0_chroma"),
        },
    },
}
