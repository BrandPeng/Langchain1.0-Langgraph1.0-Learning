# 01 - Hello LangChain: 第一个 LLM 调用

## 学习目标

通过本模块，你将学习：

1. **LangChain 1.0 的核心概念**
   - LangChain 1.0 构建在 LangGraph 运行时之上
   - 统一的模型初始化接口
   - 简化的 API 设计

2. **init_chat_model 函数**
   - 如何初始化聊天模型
   - 支持的参数和配置选项
   - 跨模型提供商的统一接口

3. **invoke 方法**
   - 同步调用模型
   - 输入格式（字符串、消息列表、字典）
   - 返回值结构

4. **Messages（消息类型）**
   - SystemMessage：系统提示
   - HumanMessage：用户输入
   - AIMessage：AI 响应

---

## 核心概念详解

### 1. init_chat_model - 模型初始化

`init_chat_model` 是 LangChain 1.0 中用于初始化聊天模型的**统一接口**。

#### 基本语法

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "provider:model_name",  # 提供商:模型名称
    api_key="your-api-key",  # API 密钥（可选，可从环境变量读取）
    temperature=0.7,         # 温度参数（可选）
    max_tokens=1000,         # 最大 token 数（可选）
    **kwargs                 # 其他模型特定参数
)
```

#### 参数详解

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | `str` | **必需**。格式为 `"provider:model_name"`，如 `"groq:llama-3.3-70b-versatile"` | 无 |
| `api_key` | `str` | API 密钥。如果不提供，会从环境变量中读取（如 `GROQ_API_KEY`） | `None` |
| `temperature` | `float` | 控制输出随机性。范围 0.0-2.0。<br>- `0.0`：最确定性<br>- `1.0`：默认，平衡<br>- `2.0`：最随机 | `1.0` |
| `max_tokens` | `int` | 限制模型输出的最大 token 数量 | 模型默认值 |
| `model_kwargs` | `dict` | 传递给底层模型的额外参数 | `{}` |

#### 支持的提供商格式

```python
# Groq
"groq:llama-3.3-70b-versatile"
"groq:llama-3.1-8b-instant"

# OpenAI
"openai:gpt-4"
"openai:gpt-3.5-turbo"

# Anthropic
"anthropic:claude-sonnet-4-5-20250929"

# 其他提供商...
```

#### 为什么使用 init_chat_model？

1. **统一接口**：无需记住每个提供商的不同初始化方式
2. **易于切换**：只需修改模型字符串即可切换模型
3. **类型安全**：提供更好的类型提示
4. **简洁明了**：减少样板代码

#### 示例

```python
from langchain.chat_models import init_chat_model
import os

# 方式 1：直接传递 API key
model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    api_key="your-groq-api-key"
)

# 方式 2：从环境变量读取（推荐）
model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 方式 3：配置温度和 token 限制
model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,    # 最确定性输出
    max_tokens=500      # 限制输出长度
)
```

---

### 2. invoke 方法 - 调用模型（深入详解）

`invoke` 是 LangChain 中**最核心的方法**，用于同步调用 LLM 模型。理解 invoke 是学习 LangChain 的关键。

---

#### 🎯 invoke 方法做什么？

简单来说，`invoke` 方法的作用就是：

1. **接收你的输入**（问题、指令、对话历史等）
2. **发送给 LLM 模型**（如 GPT-4, Llama, Claude 等）
3. **返回模型的响应**（文本回复 + 元数据信息）

**流程图：**
```
你的输入 → invoke() → LLM 模型 → 响应 → 返回给你
```

---

#### 📝 基本语法

```python
response = model.invoke(input, config=None)
```

**参数详解：**

| 参数 | 类型 | 说明 | 必需 | 默认值 |
|------|------|------|------|--------|
| `input` | `str` \| `list[dict]` \| `list[Message]` | 你要发送给模型的内容 | ✅ 必需 | 无 |
| `config` | `dict` | 高级配置（回调函数、元数据、标签等） | ❌ 可选 | `None` |

---

#### 🔍 深入理解 input 参数 - 三种输入格式

这是最容易困惑的地方！`invoke` 支持**三种不同的输入格式**，让我们逐一详解：

---

##### 📌 格式 1：纯字符串（最简单，适合单次问答）

**使用场景：** 简单的一次性问答，不需要设置系统角色或对话历史

**语法：**
```python
response = model.invoke("你的问题或指令")
```

**完整示例：**
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile", api_key="your_key")

# 直接传递字符串
response = model.invoke("什么是机器学习？用一句话解释")

print(response.content)
# 输出：机器学习是一种让计算机通过数据学习规律，而无需明确编程的技术。
```

**优点：**
- ✅ 最简单，代码最少
- ✅ 适合快速测试

**缺点：**
- ❌ 无法设置系统提示（system prompt）
- ❌ 无法传递对话历史
- ❌ 灵活性较低

**什么时候用？**
- 快速测试
- 简单的一次性问答
- 不需要上下文的场景

---

##### 📌 格式 2：字典列表（推荐，最灵活）

**使用场景：** 需要设置系统角色、多轮对话、精确控制对话流程

**语法：**
```python
messages = [
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "AI回复"},  # 可选，用于对话历史
    {"role": "user", "content": "继续提问"}
]
response = model.invoke(messages)
```

**角色说明：**

| 角色 | 英文 | 作用 | 示例 |
|------|------|------|------|
| `system` | System | 设定 AI 的行为、角色、规则 | "你是一个专业的 Python 导师" |
| `user` | Human/User | 用户的输入/问题 | "什么是装饰器？" |
| `assistant` | AI/Assistant | AI 的历史回复（用于对话上下文） | "装饰器是一种设计模式..." |

**完整示例 1：设置系统提示**
```python
messages = [
    {
        "role": "system",
        "content": "你是一个专业的 Python 编程导师。回答要简洁、准确，并提供代码示例。"
    },
    {
        "role": "user",
        "content": "什么是 Python 列表推导式？"
    }
]

response = model.invoke(messages)
print(response.content)
```

**完整示例 2：多轮对话（带历史）**
```python
# 第一轮对话
messages = [
    {"role": "system", "content": "你是一个友好的助手"},
    {"role": "user", "content": "我叫小明"}
]

response1 = model.invoke(messages)
print(response1.content)  # "你好，小明！很高兴认识你。"

# 第二轮对话 - 添加历史
messages.append({"role": "assistant", "content": response1.content})
messages.append({"role": "user", "content": "我刚才说我叫什么？"})

response2 = model.invoke(messages)
print(response2.content)  # "你说你叫小明。"
```

**完整示例 3：构建完整对话**
```python
# 初始化对话
conversation = [
    {"role": "system", "content": "你是一个 Python 专家"}
]

# 用户提问 1
conversation.append({"role": "user", "content": "什么是列表？"})
response1 = model.invoke(conversation)
print(f"AI: {response1.content}")

# 保存 AI 回复到历史
conversation.append({"role": "assistant", "content": response1.content})

# 用户提问 2（基于上下文）
conversation.append({"role": "user", "content": "它和元组有什么区别？"})
response2 = model.invoke(conversation)
print(f"AI: {response2.content}")

# 此时 conversation 包含完整的对话历史
print(f"\n完整对话历史: {conversation}")
```

**优点：**
- ✅ 最灵活，完全控制
- ✅ 可以设置系统提示
- ✅ 支持多轮对话
- ✅ 与 OpenAI API 格式一致
- ✅ JSON 兼容，易于存储和传输

**缺点：**
- ❌ 代码稍微多一点（但更清晰）

**什么时候用？**
- ✅ **推荐用于所有场景**
- 需要设置系统角色
- 多轮对话
- 需要保存对话历史
- 生产环境应用

---

##### 📌 格式 3：消息对象列表（类型安全，但较繁琐）

**使用场景：** 需要类型检查、IDE 自动补全的场景

**语法：**
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="系统提示"),
    HumanMessage(content="用户消息"),
    AIMessage(content="AI回复")
]
response = model.invoke(messages)
```

**消息类型对照：**

| 消息类 | 对应字典格式 | 作用 |
|--------|-------------|------|
| `SystemMessage` | `{"role": "system", ...}` | 系统提示 |
| `HumanMessage` | `{"role": "user", ...}` | 用户输入 |
| `AIMessage` | `{"role": "assistant", ...}` | AI 回复 |

**完整示例：**
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="你是一个 Python 专家"),
    HumanMessage(content="什么是生成器？"),
]

response = model.invoke(messages)

# 继续对话
messages.append(AIMessage(content=response.content))
messages.append(HumanMessage(content="能给个例子吗？"))

response2 = model.invoke(messages)
```

**优点：**
- ✅ 类型安全
- ✅ IDE 自动补全
- ✅ 更容易发现错误

**缺点：**
- ❌ 代码较长
- ❌ 不如字典简洁
- ❌ 难以序列化（JSON）

**什么时候用？**
- 大型项目，需要类型检查
- 团队协作，需要严格规范
- 使用 TypeScript/MyPy 等类型检查工具

---

#### 🎁 invoke 返回值详解

`invoke` 返回一个 **AIMessage 对象**，包含丰富的信息：

**返回对象结构：**
```python
response = model.invoke("Hello")

# 1. 主要内容
response.content              # str - AI 的回复文本
response.response_metadata    # dict - 响应元数据
response.id                   # str - 消息唯一 ID
response.usage_metadata       # dict - Token 使用情况
response.additional_kwargs    # dict - 其他额外信息
```

**完整示例：访问所有信息**
```python
response = model.invoke("用一句话解释什么是 AI")

# 1. 获取回复内容
print("AI 回复:", response.content)

# 2. 获取模型信息
metadata = response.response_metadata
print(f"使用的模型: {metadata['model_name']}")
print(f"结束原因: {metadata['finish_reason']}")

# 3. 获取 Token 使用情况
usage = metadata.get('token_usage', {})
print(f"提示 tokens: {usage.get('prompt_tokens')}")
print(f"完成 tokens: {usage.get('completion_tokens')}")
print(f"总计 tokens: {usage.get('total_tokens')}")

# 4. 获取消息 ID
print(f"消息 ID: {response.id}")
```

**response_metadata 完整结构：**
```python
{
    'model_name': 'llama-3.3-70b-versatile',      # 使用的模型
    'system_fingerprint': 'fp_4cfc2deea6',        # 系统指纹
    'finish_reason': 'stop',                      # 结束原因：stop/length/error
    'model_provider': 'groq',                     # 模型提供商
    'token_usage': {                              # Token 使用统计
        'prompt_tokens': 15,                      # 输入 tokens
        'completion_tokens': 25,                  # 输出 tokens
        'total_tokens': 40,                       # 总计 tokens
        'prompt_time': 0.002,                     # 输入处理时间（秒）
        'completion_time': 0.23                   # 输出生成时间（秒）
    }
}
```

---

#### 🔧 config 参数（高级用法）

`config` 参数用于传递高级配置，一般初学者不需要用到。

**常用配置：**

```python
config = {
    "callbacks": [callback_handler],      # 回调函数
    "tags": ["test", "development"],      # 标签（用于追踪）
    "metadata": {"user_id": "123"},       # 元数据
    "run_name": "my_query"                # 运行名称
}

response = model.invoke(messages, config=config)
```

**暂时可以忽略，后续会详细学习。**

---

#### 📚 实战示例汇总

**示例 1：最简单的问答**
```python
response = model.invoke("什么是 Python？")
print(response.content)
```

**示例 2：带系统提示的问答**
```python
messages = [
    {"role": "system", "content": "你是一个幽默的助手，喜欢用比喻解释概念"},
    {"role": "user", "content": "什么是递归？"}
]
response = model.invoke(messages)
print(response.content)
```

**示例 3：多轮对话**
```python
conversation = [
    {"role": "system", "content": "你是一个编程助手"}
]

# 第一轮
conversation.append({"role": "user", "content": "Python 中如何定义函数？"})
r1 = model.invoke(conversation)
conversation.append({"role": "assistant", "content": r1.content})

# 第二轮
conversation.append({"role": "user", "content": "那参数怎么传递？"})
r2 = model.invoke(conversation)
print(r2.content)
```

**示例 4：监控 Token 使用**
```python
response = model.invoke("写一首关于编程的诗")
usage = response.response_metadata['token_usage']

print(f"本次调用使用了 {usage['total_tokens']} 个 tokens")
print(f"成本预估: ${usage['total_tokens'] * 0.0001:.4f}")  # 假设每千tokens $0.1
```

---

### 3. Messages - 消息类型

LangChain 使用不同的消息类型来表示对话中的不同角色。

#### 消息类型概览

| 消息类型 | 角色 | 用途 | 示例 |
|---------|------|------|------|
| `SystemMessage` | `system` | 设定 AI 的行为、角色、规则 | "你是一个专业的数学老师" |
| `HumanMessage` | `user` | 用户的输入 | "什么是微积分？" |
| `AIMessage` | `assistant` | AI 的回复 | "微积分是研究变化率的数学分支..." |

#### 使用消息对象

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 创建消息
system_msg = SystemMessage(content="你是一个友好的助手")
human_msg = HumanMessage(content="你好")
ai_msg = AIMessage(content="你好！我能帮你什么？")

# 构建对话历史
messages = [system_msg, human_msg, ai_msg]
messages.append(HumanMessage(content="今天天气怎么样？"))

# 调用模型
response = model.invoke(messages)
```

#### 使用字典格式（推荐）

```python
# 更简洁的字典格式
messages = [
    {"role": "system", "content": "你是一个友好的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我能帮你什么？"},
    {"role": "user", "content": "今天天气怎么样？"}
]

response = model.invoke(messages)
```

#### 为什么使用不同的消息类型？

1. **明确角色**：清晰区分系统提示、用户输入和 AI 回复
2. **对话历史**：构建完整的多轮对话上下文
3. **控制行为**：通过 SystemMessage 精确控制 AI 的行为
4. **调试友好**：更容易追踪和调试对话流程

---

## 完整示例代码说明

`main.py` 文件包含 7 个渐进式示例：

### 示例 1：最简单的 LLM 调用
- 演示基本的 `init_chat_model` 和 `invoke` 使用
- 使用字符串作为输入

### 示例 2：使用消息列表进行对话
- 使用 `SystemMessage` 和 `HumanMessage`
- 构建多轮对话历史

### 示例 3：使用字典格式的消息（推荐）
- 更简洁的字典格式
- 与 OpenAI API 格式一致

### 示例 4：配置模型参数
- 对比不同 `temperature` 的效果
- 使用 `max_tokens` 限制输出

### 示例 5：理解 invoke 返回值
- 详细解析 `AIMessage` 对象
- 访问元数据和 token 使用情况

### 示例 6：错误处理
- 捕获和处理常见错误
- 生产环境的最佳实践

### 示例 7：多模型对比
- 轻松切换不同模型
- 对比不同模型的输出

---

## 环境配置

### 1. 安装依赖

```bash
pip install langchain langchain-groq python-dotenv
```

### 2. 配置 API 密钥

创建 `.env` 文件：

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. 获取 Groq API 密钥

1. 访问 [Groq Console](https://console.groq.com/)
2. 注册账号
3. 在 API Keys 页面创建新的 API 密钥
4. 复制密钥到 `.env` 文件

---

## 运行示例

```bash
# 在项目根目录
cd phase1_fundamentals/01_hello_langchain

# 运行所有示例
python main.py
```

---

## 常见问题 (FAQ)

### Q1: init_chat_model 和直接使用 ChatGroq 有什么区别？

**A:** `init_chat_model` 是 LangChain 1.0 的统一接口，优势包括：
- 跨模型提供商的一致 API
- 更简洁的语法
- 更好的类型提示
- 更容易切换模型

```python
# 旧方式（仍然可用）
from langchain_groq import ChatGroq
model = ChatGroq(model="llama-3.3-70b-versatile", api_key="...")

# 新方式（推荐）
from langchain.chat_models import init_chat_model
model = init_chat_model("groq:llama-3.3-70b-versatile", api_key="...")
```

### Q2: temperature 参数如何选择？

**A:** 根据使用场景选择：
- **0.0-0.3**：需要一致性、准确性的任务（数据提取、分类、代码生成）
- **0.5-0.7**：平衡创造性和一致性（聊天、问答）
- **0.8-1.5**：创造性任务（写作、头脑风暴）
- **1.5-2.0**：高度创造性（诗歌、故事创作）

### Q3: invoke 和 stream 有什么区别？

**A:**
- `invoke`：同步调用，等待完整响应后返回
- `stream`：流式调用，实时返回响应片段（我们将在后续模块学习）

```python
# invoke - 等待完整响应
response = model.invoke("写一首诗")
print(response.content)  # 一次性输出完整诗歌

# stream - 实时流式输出（后续学习）
for chunk in model.stream("写一首诗"):
    print(chunk.content, end="", flush=True)  # 逐字输出
```

### Q4: 为什么推荐使用字典格式而不是消息对象？

**A:** 两种方式都可以，但字典格式有以下优势：
- 更简洁，代码量更少
- 与 OpenAI API 格式一致
- 更容易序列化和存储
- JSON 兼容，便于网络传输

### Q5: 如何处理 API 调用失败？

**A:** 使用 try-except 块捕获异常：

```python
try:
    response = model.invoke("Hello")
    print(response.content)
except ValueError as e:
    print(f"配置错误: {e}")
except ConnectionError as e:
    print(f"网络错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

---

## 最佳实践

### 1. 使用环境变量管理 API 密钥

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

### 2. 验证 API 密钥存在

```python
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("请设置 GROQ_API_KEY 环境变量!")
```

### 3. 使用 SystemMessage 控制行为

```python
messages = [
    {"role": "system", "content": "你是一个专业且简洁的助手。回答限制在100字以内。"},
    {"role": "user", "content": "什么是机器学习？"}
]
```

### 4. 保存和重用对话历史

```python
# 初始化对话历史
conversation = [
    {"role": "system", "content": "你是一个友好的助手"}
]

# 添加用户消息
conversation.append({"role": "user", "content": "你好"})

# 调用模型
response = model.invoke(conversation)

# 保存 AI 回复到历史
conversation.append({"role": "assistant", "content": response.content})

# 继续对话
conversation.append({"role": "user", "content": "我想学 Python"})
response = model.invoke(conversation)
```

### 5. 监控 Token 使用

```python
response = model.invoke("Hello")
usage = response.response_metadata.get("token_usage", {})
print(f"Token 使用: {usage.get('total_tokens', 'N/A')}")
```

---

## LangChain 1.0 重要变更

### 从 0.x 到 1.0 的主要变化

1. **统一的模型初始化**
   - 旧：使用特定的类（如 `ChatGroq`, `ChatOpenAI`）
   - 新：使用 `init_chat_model` 统一接口

2. **简化的 Agent 创建**
   - 旧：使用 `initialize_agent` 等已弃用函数
   - 新：使用 `create_agent` 统一接口（我们将在模块 5 学习）

3. **LangGraph 作为运行时**
   - LangChain 1.0 构建在 LangGraph 之上
   - 更强大的状态管理和工作流控制

4. **中间件系统**
   - 新增中间件架构（我们将在模块 10-12 学习）
   - 更好的可观测性和控制流

---

## 下一步学习

完成本模块后，继续学习：

1. **02_prompt_templates** - 学习提示词模板，避免字符串拼接
2. **03_messages** - 深入理解消息类型和对话管理
3. **04_custom_tools** - 创建自定义工具
4. **05_simple_agent** - 使用 `create_agent` 构建第一个 Agent

---

## 参考资源

- [LangChain 1.0 官方文档](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangChain 1.0 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [Groq 文档](https://console.groq.com/docs)
- [LangChain API 参考](https://docs.langchain.com/oss/python/api_reference/)

---

## 小结

通过本模块，你已经学习了：

- ✅ 如何使用 `init_chat_model` 初始化聊天模型
- ✅ 如何使用 `invoke` 方法调用模型
- ✅ 理解不同的消息类型（System, Human, AI）
- ✅ 配置模型参数（temperature, max_tokens）
- ✅ 处理响应和元数据
- ✅ 错误处理最佳实践

**恭喜！你已经迈出了 LangChain 1.0 学习的第一步！** 🎉
