from mem0 import Memory
from memory_config import config


m = Memory.from_config(config)

# 存记忆：给一段对话，Mem0 自动提取关键事实
messages = [
    {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
    {"role": "assistant", "content": "Hey Alex! I'll remember your interests."}
]
m.add(messages, user_id="alex")

# 检索记忆
results = m.search("What does Alex like?", filters={"user_id": "alex"}, top_k=3)
print(results)
