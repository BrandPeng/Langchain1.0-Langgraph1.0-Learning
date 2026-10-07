# 10 - Middleware Basics (中间件基础)

## 核心概念

**Middleware（中间件）= Agent 执行过程中的钩子函数**

在 LangChain 1.0 中，中间件是处理 Agent 生命周期的标准方式。

## 基本用法

### 创建自定义中间件

```python
from langchain.agents.middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        """模型调用前执行"""
        print("准备调用模型")
        return None  # 返回 None 表示继续正常流程

    def after_model(self, state, runtime):
        """模型响应后执行"""
        print("模型已响应")
        return None  # 返回 None 表示不修改状态

# 使用中间件
agent = create_agent(
    model=model,
    tools=[],
    middleware=[MyMiddleware()]
)
```

## 核心钩子方法

### 1. before_model（模型调用前）

```python
def before_model(self, state, runtime):
    """
    在模型调用前执行

    返回值：
    - None: 继续正常流程
    - dict: 更新状态（如 {"messages": [...]}）
    - {"jump_to": "..."}: 跳过正常流程
    """
    messages = state.get('messages', [])
    print(f"当前消息数: {len(messages)}")
    return None
```

**用途**：
- 消息修剪（trim messages）
- PII 脱敏
- 输入验证
- 条件路由

### 2. after_model（模型响应后）

```python
def after_model(self, state, runtime):
    """
    在模型响应后执行

    返回值：
    - None: 不修改状态
    - dict: 更新状态
    """
    # 统计调用次数
    count = state.get("call_count", 0)
    return {"call_count": count + 1}
```

**用途**：
- 输出验证
- 格式化响应
- 统计信息
- 状态更新

## 返回值的作用

### 返回 None
```python
def before_model(self, state, runtime):
    print("日志记录")
    return None  # 不做任何修改，继续流程
```

### 返回字典（更新状态）
```python
def after_model(self, state, runtime):
    count = state.get("count", 0)
    return {"count": count + 1}  # 更新状态中的 count
```

### 返回 jump_to（控制流程）
```python
def before_model(self, state, runtime):
    if state.get("count", 0) > 10:
        return {"jump_to": "__end__"}  # 跳过模型，直接结束
    return None
```

**jump_to 目标**：
- `"__end__"` - 结束 Agent
- `"tools"` - 跳到工具节点
- 其他自定义节点

## 执行顺序（重要！）

```python
agent = create_agent(
    model=model,
    middleware=[Middleware1(), Middleware2(), Middleware3()]
)
```

**执行流程**：
```
1. Middleware1.before_model   ↓ 正序
2. Middleware2.before_model   ↓
3. Middleware3.before_model   ↓

   [模型调用]

6. Middleware3.after_model    ↑ 逆序
5. Middleware2.after_model    ↑
4. Middleware1.after_model    ↑
```

**类似洋葱模型**：外层先进后出

## 实际应用

### 1. 日志中间件

```python
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"[日志] 消息数: {len(state.get('messages', []))}")
        return None

    def after_model(self, state, runtime):
        last_msg = state.get('messages', [])[-1]
        print(f"[日志] 响应类型: {last_msg.__class__.__name__}")
        return None
```

### 2. 计数中间件

```python
class CallCounterMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        count = state.get("model_call_count", 0)
        return {"model_call_count": count + 1}

# 需要 checkpointer 来保存自定义状态
agent = create_agent(
    model=model,
    middleware=[CallCounterMiddleware()],
    checkpointer=InMemorySaver()
)
```

### 3. 消息修剪中间件

`messages` 使用合并规则，返回列表切片不会删除旧消息。需要先用
`RemoveMessage(id=REMOVE_ALL_MESSAGES)` 清空历史，再写入要保留的消息。
这个更新在有无 checkpointer 时均有效。

```python
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

class MessageTrimmerMiddleware(AgentMiddleware):
    def __init__(self, max_messages=5):
        super().__init__()
        if max_messages < 1:
            raise ValueError("max_messages 必须大于 0")
        self.max_messages = max_messages

    def before_model(self, state, runtime):
        messages = state.get('messages', [])
        if len(messages) > self.max_messages:
            return {"messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *messages[-self.max_messages:],
            ]}
        return None
```

`max_messages=4` 限制模型调用前的历史消息数，不包括 `system_prompt`；
新增一条 AI 回复后，返回状态最多有 5 条消息。示例 3 的六轮调用后消息数为
`2、4、5、5、5、5`。

此处只演示无工具文本对话的数量限制。用于其他场景时，还需按模型提供商要求
保留系统消息和合法的对话起点，并确保工具调用与工具结果成对保留。
参见 [LangChain 消息删除说明](https://docs.langchain.com/oss/python/langchain/short-term-memory#delete-messages)。

### 4. 输出验证中间件

```python
class OutputValidationMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        last_msg = state.get('messages', [])[-1]
        content = getattr(last_msg, 'content', '')

        if len(content) > 1000:
            print("[警告] 响应过长")

        return None
```

### 5. 限流中间件

```python
class MaxCallsMiddleware(AgentMiddleware):
    def __init__(self, max_calls=10):
        super().__init__()
        self.max_calls = max_calls

    def before_model(self, state, runtime):
        count = state.get("call_count", 0)
        if count >= self.max_calls:
            return {"jump_to": "__end__"}  # 达到限制，直接结束
        return None

    def after_model(self, state, runtime):
        count = state.get("call_count", 0)
        return {"call_count": count + 1}
```

## 内置中间件

### SummarizationMiddleware（自动摘要）

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.1-8b-instant",  # 可用便宜模型
            max_tokens_before_summary=500
        )
    ],
    checkpointer=InMemorySaver()
)
```

**作用**：
- 消息超过 token 限制时自动摘要
- 保留最近消息 + 旧消息摘要
- 详见 08_context_management 章节

### HumanInTheLoopMiddleware（人工审核）

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=model,
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True}  # 调用此工具前暂停
        )
    ]
)
```

### PIIMiddleware（敏感信息处理）

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=model,
    middleware=[
        PIIMiddleware("email", strategy="redact"),      # 邮箱脱敏
        PIIMiddleware("phone_number", strategy="block") # 电话拦截
    ]
)
```

## 常见问题

### 1. 中间件能访问工具调用吗？

不能直接访问。`before_model` 和 `after_model` 只在模型节点执行。

如果需要拦截工具调用，使用 `wrap_tool_call`（高级特性）。

### 2. 多个中间件的顺序重要吗？

**非常重要！**

```python
middleware=[
    TrimmerMiddleware(),     # 1. 先修剪消息
    SummarizationMiddleware(), # 2. 再摘要
    LoggingMiddleware()      # 3. 最后记录日志
]
```

- `before_model` 按列表顺序执行
- `after_model` 按列表逆序执行

### 3. 修改状态需要 checkpointer 吗？

**自定义状态需要，messages 不需要**：

```python
# 不需要 checkpointer（messages 自动保存）
def after_model(self, state, runtime):
    return {"messages": [...]}

# 需要 checkpointer（自定义字段）
def after_model(self, state, runtime):
    return {"my_custom_field": 123}
```

### 4. 能在中间件里调用另一个模型吗？

可以，但要小心：

```python
class ValidationMiddleware(AgentMiddleware):
    def __init__(self):
        self.validator_model = init_chat_model(...)

    def after_model(self, state, runtime):
        # 用另一个模型验证输出
        last_msg = state['messages'][-1]
        validation_result = self.validator_model.invoke(...)
        return None
```

## 最佳实践

```python
# 1. 生产环境推荐配置
agent = create_agent(
    model=model,
    tools=[...],
    middleware=[
        MessageTrimmerMiddleware(max_messages=20),  # 限制消息数
        SummarizationMiddleware(model=..., max_tokens=2000), # 自动摘要
        LoggingMiddleware(),  # 日志记录
    ],
    checkpointer=SqliteSaver.from_conn_string("...")
)

# 2. 开发环境
agent = create_agent(
    model=model,
    tools=[...],
    middleware=[
        LoggingMiddleware(),  # 只要日志
    ]
)

# 3. 测试环境
agent = create_agent(
    model=model,
    tools=[...],
    middleware=[
        MaxCallsMiddleware(max_calls=5),  # 防止测试费用爆炸
    ]
)
```

## 核心要点

1. **中间件** = Agent 生命周期钩子
2. **before_model** - 模型调用前（正序执行）
3. **after_model** - 模型响应后（逆序执行）
4. **返回 None** - 不修改状态
5. **返回 dict** - 更新状态
6. **返回 {"jump_to": "..."}** - 控制流程
7. **顺序重要** - 类似洋葱模型
8. **内置中间件** - SummarizationMiddleware 最常用

## 下一步

**11_structured_output** - 学习如何使用 Pydantic 获取结构化输出
