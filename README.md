# 🦜🔗 LangChain 1.0 & LangGraph 1.0 完整学习指南

> 这是一个系统学习 **LangChain 1.0** 和 **LangGraph 1.0** 的实践仓库，涵盖从基础概念到实战项目的完整学习路径。

![img.png](img.png)

---

## 📚 关于 LangChain 1.0

LangChain 1.0 是用于构建 LLM 驱动应用程序的框架的最新主要版本（2025年10月正式发布）。主要特性：

- ✅ **构建在 LangGraph 运行时之上** - 提供持久化、流式处理、人在回路等能力
- ✅ **新的 `create_agent` API** - 简化 Agent 创建流程（LangChain 1.0 API）
- ✅ **中间件架构** - 提供细粒度的执行控制（before_model、after_model、wrap_model_call 等）
- ✅ **多模态支持** - 处理文本、图像、视频、文件
- ✅ **结构化输出** - 使用 Pydantic 模型定义输出格式
- ✅ **语义化版本控制** - 1.x 系列保证 API 稳定

**后续计划** ： 
1. 基于Langchain 1.0 的完整RAG前后端项目，并包含RAG的各种优化   
⚫ 技术栈：Agentic RAG、Agent、LangChain、ChromaDB、SQLite、FastAPI、Vue。   
2. 基于LangGrah 1.0的完整多智能体前后端项目。   
⚫ 技术栈：：Python、LangGraph、LangChain、FastAPI、MCP、Vue、SQLite
---

## 🚀 快速开始

### 环境要求

- **Python 3.10 或更高版本**（不支持 Python 3.9）
- pip 或 uv 包管理器

### 安装步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd Langchain1.0-Langgraph1.0-learning

# 2. 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Keys
```

### 需要的 API Keys

| API Key | 用途 | 获取地址 |
|---------|------|----------|
| `GROQ_API_KEY` | Groq API（免费） | https://console.groq.com/keys |
| `OPENAI_API_KEY` | OpenAI API（可选） | https://platform.openai.com/api-keys |
| `PINECONE_API_KEY` | Pinecone 向量数据库（免费） | https://www.pinecone.io/ |
| `LANGSMITH_API_KEY` | LangSmith 监控（可选） | https://smith.langchain.com/ |

### 验证安装

```bash
python phase1_fundamentals/01_hello_langchain/main.py
```

---

## 📖 学习路径

本仓库采用**四阶段渐进式学习**，共 22 个模块 + 3 个综合项目：

```
📚 完整学习路径
================

第一阶段：基础知识 (Phase 1 - Fundamentals) - 第1-2周
├── 01 Hello LangChain - 第一次 LLM 调用
├── 02 Prompt Templates - 提示词模板
├── 03 Messages - 消息类型与对话历史
├── 04 Custom Tools - 自定义工具
├── 05 Simple Agent - create_agent 入门
└── 06 Agent Loop - Agent 执行循环

第二阶段：实战技能 (Phase 2 - Practical) - 第3-4周
├── 07 Memory Basics - 内存基础
├── 08 Context Management - 上下文管理
├── 09 Checkpointing - 状态持久化
├── 10 Middleware Basics - 中间件基础
├── 11 Structured Output - 结构化输出
├── 12 Validation Retry - 验证与重试
├── 13 RAG Basics - RAG 基础
├── 14 RAG Advanced - RAG 进阶
└── 15 Tools and Agents - 工具与智能体进阶

第三阶段：高级主题 (Phase 3 - Advanced) - 第5-6周
├── 16 LangGraph Basics - 状态图基础
├── 17 Multi-Agent - 多智能体系统
├── 18 Conditional Routing - 条件路由
├── 19 Image Input - 图像输入处理
├── 20 File Handling - 文件处理
├── 21 Mixed Modality - 混合模态
├── 22 LangSmith Integration - 监控集成
└── 23 Error Handling - 错误处理

第四阶段：综合项目 (Phase 4 - Projects) - 第7-8周
├── 01 RAG System - 检索增强生成系统
├── 02 Multi-Agent Support - 多智能体客服系统
└── 03 Research Assistant - 智能研究助手
```

---

## 📁 项目结构

```
Langchain1.0-Langgraph1.0-learning/
├── phase1_fundamentals/        # 第一阶段：基础知识
│   ├── 01_hello_langchain/     # 第一次 LLM 调用
│   ├── 02_prompt_templates/    # 提示词模板
│   ├── 03_messages/            # 消息类型
│   ├── 04_custom_tools/        # 自定义工具
│   ├── 05_simple_agent/        # 简单 Agent
│   └── 06_agent_loop/          # Agent 执行循环
│
├── phase2_practical/           # 第二阶段：实战技能
│   ├── 07_memory_basics/       # 内存基础
│   ├── 08_context_management/  # 上下文管理
│   ├── 09_checkpointing/       # 状态持久化
│   ├── 10_middleware_basics/   # 中间件基础
│   ├── 11_structured_output/   # 结构化输出
│   ├── 12_validation_retry/    # 验证与重试
│   ├── 13_rag_basics/          # RAG 基础
│   ├── 14_rag_advanced/        # RAG 进阶
│   └── 15_tools_and_agents/    # 工具与智能体进阶
│
├── phase3_advanced/            # 第三阶段：高级主题
│   ├── 16_langgraph_basics/    # LangGraph 基础
│   ├── 17_multi_agent/         # 多智能体系统
│   ├── 18_conditional_routing/ # 条件路由
│   ├── 19_image_input/         # 图像输入
│   ├── 20_file_handling/       # 文件处理
│   ├── 21_mixed_modality/      # 混合模态
│   ├── 22_langsmith_integration/ # LangSmith 集成
│   └── 23_error_handling/      # 错误处理
│
├── phase4_projects/            # 第四阶段：综合项目
│   ├── 01_rag_system/          # RAG 系统
│   ├── 02_multi_agent_support/ # 多智能体客服
│   └── 03_research_assistant/  # 研究助手
│
├── docs/                       # 学习文档
├── .env.example                # 环境变量模板
├── requirements.txt            # 依赖列表
└── README.md                   # 本文件
```

---

## 🎯 各阶段详细内容

### 第一阶段：基础知识

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 01 | Hello LangChain | `init_chat_model`、`invoke` 方法、环境配置 |
| 02 | Prompt Templates | 文本模板、对话模板、变量替换、LCEL |
| 03 | Messages | HumanMessage、AIMessage、SystemMessage、对话历史 |
| 04 | Custom Tools | `@tool` 装饰器、参数类型、docstring 重要性 |
| 05 | Simple Agent | `create_agent` 创建 Agent、配置选项 |
| 06 | Agent Loop | 执行循环、消息历史、流式输出 |

**核心代码示例：**

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # LangChain 1.0 正确的 API
from langchain_core.tools import tool

# 初始化模型
model = init_chat_model("groq:llama-3.3-70b-versatile")

# 创建工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    return f"{city}: 晴，25°C"

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个天气助手"  # LangChain 1.0 使用 system_prompt 参数
)

# 运行
response = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气怎么样？"}]
})
```

### 第二阶段：实战技能

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 07 | Memory Basics | InMemorySaver、会话管理 |
| 08 | Context Management | 消息修剪、上下文窗口 |
| 09 | Checkpointing | SQLite 持久化、状态恢复 |
| 10 | Middleware Basics | 自定义中间件、钩子函数 |
| 11 | Structured Output | Pydantic 模型、输出解析 |
| 12 | Validation Retry | 验证失败处理、重试机制 |
| 13 | RAG Basics | 文档加载、分块、向量存储、检索 |
| 14 | RAG Advanced | 混合搜索、重排序、高级检索 |

### 第三阶段：高级主题

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 16 | LangGraph Basics | StateGraph、节点、边、检查点 |
| 17 | Multi-Agent | Supervisor 模式、协作调度 |
| 18 | Conditional Routing | 动态分支、决策树 |
| 19 | Image Input | 视觉理解、图像分析 |
| 20 | File Handling | 文档加载、多格式支持 |
| 21 | Mixed Modality | 文本+图像+数据综合处理 |
| 22 | LangSmith Integration | 追踪、监控、性能分析 |
| 23 | Error Handling | 重试、降级、容错机制 |

### 第四阶段：综合项目

| 项目 | 描述 | 核心技术 |
|------|------|----------|
| RAG System | 文档问答系统 | 向量存储、检索增强生成、引用追踪 |
| Multi-Agent Support | 智能客服系统 | 多 Agent 协作、意图识别、路由分发 |
| Research Assistant | 研究助手 | 多阶段工作流、报告生成、引用管理 |

---

## 🔧 运行示例

### 运行单个模块

```bash
# 进入模块目录
cd phase1_fundamentals/01_hello_langchain

# 运行主程序
python main.py

# 运行测试（如果有）
python test.py
```

### 运行综合项目

```bash
# 进入项目目录
cd phase4_projects/01_rag_system

# 运行项目
python main.py
```

---

## 💡 核心知识点总结

### 1. LangChain 1.0 模型调用

```python
from langchain.chat_models import init_chat_model

# 三种输入格式
model = init_chat_model("groq:llama-3.3-70b-versatile")
model.invoke("简单文本")
model.invoke([{"role": "user", "content": "字典格式"}])
model.invoke([HumanMessage("消息对象")])
```

### 2. 创建工具

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具描述 - AI 读这个来理解何时使用此工具！"""
    return "result"
```

### 3. 创建 Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[tool1, tool2],
    system_prompt="Agent 的行为指令"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "问题"}]
})
```

> ⚠️ **注意**：`langgraph.prebuilt.create_react_agent` 已弃用，将在 V2.0 移除。
> 请统一使用 `langchain.agents.create_agent`。

### 4. LangGraph 状态图（高级定制）

当 `create_agent` 无法满足需求时，使用底层 API：

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()
```

---

## ❓ 常见问题

### 1. API 密钥问题

确保 `.env` 文件配置正确：

```bash
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...  # 可选
```

### 2. 导入错误

LangChain 1.0 正确的导入路径：

```python
# 模型
from langchain.chat_models import init_chat_model

# Agent
from langchain.agents import create_agent  # ✅ 推荐

# 工具
from langchain_core.tools import tool

# LangGraph（高级定制）
from langgraph.graph import StateGraph, START, END
```

> ⚠️ `from langgraph.prebuilt import create_react_agent` 已弃用

### 3. Agent 不调用工具

- 检查工具的 docstring 是否清晰
- 确保问题明确需要该工具
- 工具参数类型注解完整

### 4. 对话不记忆

必须传入完整历史：

```python
# ❌ 错误
model.invoke("你记得我的名字吗？")

# ✅ 正确
conversation = [previous_messages...] + [new_message]
model.invoke(conversation)
```

---

## 📚 重要资源

- **LangChain 官方文档**: https://docs.langchain.com/oss/python/langchain/
- **LangGraph 文档**: https://docs.langchain.com/oss/python/langgraph
- **迁移指南**: https://docs.langchain.com/oss/python/migrate/langchain-v1
- **LangSmith 平台**: https://smith.langchain.com
- **GitHub 仓库**: https://github.com/langchain-ai/langchain

---

## 📄 许可证

MIT License

---

## 🎓 致谢

- https://github.com/Mason-zy/Langchain1.0-Study
- LangChain 官方团队

---

**开始学习之旅** 👉 `cd phase1_fundamentals/01_hello_langchain && python main.py`
