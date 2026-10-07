"""
简单测试：验证 Agent 执行循环
"""

import os
import sys

# 添加工具目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from calculator import calculator

# 加载环境变量
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 GROQ_API_KEY\n"
        "访问 https://console.groq.com/keys 获取免费密钥"
    )

# 初始化模型
model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)



print("=" * 70)
print("测试：Agent 执行循环")
print("=" * 70)

agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt="你是一个有帮助的助手，可以使用计算器进行数学计算。"
)

print("\n问题：10 加 20 等于多少？")
response = agent.invoke({
    "messages": [{"role": "user", "content": "10 加 20 等于多少？"}]
})

print("\n完整消息历史：")
for i, msg in enumerate(response['messages'], 1):
    msg_type = msg.__class__.__name__
    print(f"\n消息 {i}: {msg_type}")

    if hasattr(msg, 'content') and msg.content:
        print(f"  内容: {msg.content}")

    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  工具调用: {msg.tool_calls[0]['name']}")

print("\n" + "=" * 70)
print("最终答案:", response['messages'][-1].content)
print("=" * 70)

# 测试流式输出
print("\n测试流式输出：")
print("问题：5 乘以 6")
print("-" * 70)

for chunk in agent.stream({
    "messages": [{"role": "user", "content": "5 乘以 6"}]
}, stream_mode="values"):
    if 'messages' in chunk:
        latest = chunk['messages'][-1]
        if latest.type == 'ai' and latest.content:
            if not latest.tool_calls:
                print(f"最终答案: {latest.content}")

print("\n测试成功！")
