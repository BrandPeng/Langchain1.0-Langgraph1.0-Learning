"""Offline regressions for issues #1, #4 and #6 (no API keys required)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import dotenv
import langchain.chat_models
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

ROOT = Path(__file__).resolve().parents[1]


class RecordingModel(FakeMessagesListChatModel):
    calls: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        result = super()._generate(messages, **kwargs)
        # Each invocation is a new response, even when cycling through fixtures.
        result.generations[0].message = result.generations[0].message.model_copy(
            deep=True, update={"id": None}
        )
        return result


@pytest.fixture
def load_example(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        langchain.chat_models, "init_chat_model",
        lambda *args, **kwargs: RecordingModel(responses=[AIMessage(content="好的。")]),
    )
    old_path = sys.path[:]

    def load(relative_path):
        spec = importlib.util.spec_from_file_location("example_under_test", ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    sys.path[:] = old_path


def test_multiple_models_invokes_each_selected_model(load_example, monkeypatch):
    module = load_example("phase1_fundamentals/01_hello_langchain/main.py")
    models = []

    def initialize(name, **kwargs):
        selected = Mock()
        selected.invoke.return_value = AIMessage(content=name)
        models.append((name, selected))
        return selected

    monkeypatch.setattr(module, "init_chat_model", initialize)
    module.example_7_multiple_models()
    assert len(models) == 2
    assert len({name for name, _ in models}) == 2
    for _, selected in models:
        selected.invoke.assert_called_once_with("用一句话解释什么是机器学习。")
    assert module.model.calls == []


def test_inspect_state_prints_model_tool_and_answer(load_example, capsys):
    module = load_example("phase1_fundamentals/06_agent_loop/main.py")
    module.model = RecordingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "calculator", "args": {"operation": "divide", "a": 100, "b": 5},
            "id": "call_calculator", "type": "tool_call",
        }]),
        AIMessage(content="答案是 20。"),
    ])
    module.example_4_inspect_state()
    output = capsys.readouterr().out
    assert "工具调用: calculator" in output
    assert "100.0 divide 5.0 = 20.0" in output
    assert "答案是 20。" in output
    assert "节点: model" in output
    assert "节点: tools" in output


@pytest.mark.parametrize("with_checkpointer", [False, True])
@pytest.mark.parametrize("limit", [1, 4, 5])
def test_trimming_removes_history_before_model_and_from_state(load_example, with_checkpointer, limit):
    module = load_example("phase2_practical/10_middleware_basics/main.py")
    middleware = module.MessageTrimmerMiddleware(max_messages=limit)
    model = module.model
    agent = create_agent(
        model=model, tools=[], system_prompt="简短回复。", middleware=[middleware],
        checkpointer=InMemorySaver() if with_checkpointer else None,
    )
    config = {"configurable": {"thread_id": "trimming"}}
    history = []
    full_history = []
    for index in range(6):
        user = HumanMessage(content=f"消息{index + 1}")
        history.append(user)
        full_history.append(user.content)
        result = agent.invoke({"messages": [user] if with_checkpointer else history}, config)
        actual_input = [m for m in model.calls[-1] if not isinstance(m, SystemMessage)]
        assert [m.content for m in actual_input] == full_history[-limit:]
        assert len(actual_input) <= limit
        history = result["messages"]
        assert [m.content for m in history] == full_history[-limit:] + ["好的。"]
        assert len(history) <= limit + 1
        if with_checkpointer:
            assert agent.get_state(config).values["messages"] == history
        full_history.append("好的。")
    assert middleware.trimmed_count > 0


def test_streaming_only_labels_ai_answer_as_final(load_example, capsys):
    module = load_example("phase1_fundamentals/06_agent_loop/main.py")
    module.model = RecordingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "get_weather", "args": {"city": "北京"},
            "id": "call_weather", "type": "tool_call",
        }]),
        AIMessage(content="天气查询完成。"),
    ])
    module.example_2_streaming()
    output = capsys.readouterr().out
    assert output.count("最终回答:") == 1
    assert "最终回答: 天气查询完成。" in output


@pytest.mark.parametrize("limit", [0, -1])
def test_trimmer_rejects_nonpositive_limits(load_example, limit):
    module = load_example("phase2_practical/10_middleware_basics/main.py")
    with pytest.raises(ValueError, match="max_messages"):
        module.MessageTrimmerMiddleware(max_messages=limit)
