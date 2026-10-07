# 离线回归测试

在 Python 3.10+ 虚拟环境中，从仓库根目录运行：

```bash
python -m pip install 'langchain>=1.0,<2' 'langgraph>=1.0,<2' python-dotenv pytest
python -m pytest tests/test_issue_regressions.py -q
```

测试以本地假模型替代模型服务，运行真实的 Agent 图、工具和消息状态更新，
无需 API 密钥或网络。覆盖模型切换（#4）、流式中间状态（#6）及多轮消息
修剪（#1，包含有无 checkpointer 的情况）。现有各章节 `test.py` 属于需要
模型 API 的演示脚本，不在此离线测试命令的收集范围内。
