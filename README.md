# RAG Agent FastAPI

一个基于 FastAPI、LangGraph 与 MCP 的课程资料智能问答 Agent。它把独立 RAG 服务和学习计划能力封装为 MCP 工具，并支持可控记忆、多课程学习计划、工具保护、多模型降级与 SSE 流式输出。

## 核心能力

- 通过 HTTP 调用独立部署的 RAG 服务检索课程资料
- 使用显式 LangGraph `StateGraph` 实现“模型判断 → MCP 工具执行 → 再判断”的可控闭环
- Agent 自动选择五个 MCP 工具：
  - `search_course_knowledge`：检索课程资料
  - `create_or_update_study_plan`：按课程创建或更新学习计划
  - `get_study_plan` / `list_study_plans`：读取指定课程计划或计划列表
  - `save_learning_preferences`：仅保存用户明确表达的稳定学习偏好
- Redis 缓存 RAG 工具的问答结果，并保存短期会话记忆、显式偏好与多课程计划；相同问题命中缓存时，Agent 仍判断工具，但跳过 RAG HTTP 请求
- 请求级工具保护：参数校验、重复工具调用拦截、最大调用次数限制、结构化成功/失败结果
- `qwen-flash` 主模型，`qwen-plus` 失败降级
- SSE 流式响应，实时返回“正在检索资料”等状态
- RAG 请求重试、超时处理与调用链 `trace_id`
- 透传 RAG V2 的来源文件与调用链，最终回答可追溯资料依据
- Agent JSONL 调用日志：模型切换、工具/拦截记录、记忆加载、缓存、来源、耗时与 RAG 调用链
- Docker Compose 启动 Agent 与 Redis；Agent 通过 HTTP 调用独立运行的 RAG 服务

## 技术栈

Python、FastAPI、LangChain、LangGraph、MCP、通义千问、Redis、SSE、Docker、Docker Compose。

## 架构

前端/调用方 → FastAPI Agent → LangGraph 判断节点 → MCP 工具节点 → RAG 服务或学习计划存储 → LangGraph 最终回答。

`agent/graph.py` 只定义图状态、节点与边；`agent/runtime.py` 负责组装消息并执行同步或流式图；`agent/tools.py` 是 LangGraph 可见的工具代理；`agent/mcp_server.py` 承担真正的工具业务实现。

RAG 服务是独立项目 `RAG_FastAPI`，Agent 通过 MCP 工具内的 `RagApiClient` 访问它，而不是直接耦合 RAG 源码。Agent 将自己的 `X-Trace-ID` 传给 RAG；RAG 返回的 `sources` 会由 Agent 透传给调用方。Agent 调用 RAG 时关闭 RAG 自身的会话历史，避免与 Agent 的 Redis 记忆重复。

## 本地配置

复制示例配置：

```bash
cp .env.example .env
```

然后在 `.env` 中填写自己的：

```env
DASHSCOPE_API_KEY=你的百炼API密钥
AGENT_MODEL_NAME=qwen-flash
AGENT_FALLBACK_MODEL_NAME=qwen-plus
AGENT_SHORT_MEMORY_MAX_TURNS=4
AGENT_MAX_TOOL_CALLS=6
```

不要上传 `.env`。应填写归属项目业务空间的 API Key；不要使用聊天记录中已经暴露过的旧 Key。

配置优先级为：环境变量 > `.env` > 代码默认值。`Settings` 读取密钥后显式传给主、备模型；不需要把 `.env` 中的密钥另外导出为环境变量。密钥没有默认值，使用模型前必须提供。

直接运行时，先准备好独立的 RAG 和 Redis 服务，再从本项目根目录执行：

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

`.env` 按当前工作目录查找，因此应在本项目根目录启动。Docker Compose 则将配置注入容器环境，并覆盖容器内的 RAG、Redis 地址。

## Docker 启动

先在 `RAG_FastAPI` 项目启动唯一的 RAG 服务（`http://127.0.0.1:8000`）；
本项目的 `docker-compose.yml` 只启动 Redis 与 Agent（`http://127.0.0.1:8001`）。

```bash
docker compose up --build -d
```

查看运行状态：

```bash
docker compose ps
```

接口文档：

- Agent：http://127.0.0.1:8001/docs

停止服务：

```bash
docker compose down
```

## Agent 接口

```http
POST /api/agent/chat
```

请求示例：

```json
{
  "question": "INFS7410 的 Weekly quizzes 如何计分？",
  "session_id": "demo-session"
}
```

响应会返回 `answer`、`session_id`、`trace_id` 与 `sources`。流式接口的最后一个 `done` 事件同样包含来源文件与 RAG 调用链 ID。

流式接口：

```http
POST /api/agent/chat/stream
```

## 测试与测评

运行单元测试：

```bash
python -B -m unittest discover -s tests -v
```

运行真实测评（逐题隔离，单题超过 90 秒记为失败并继续）：

```bash
EVALUATION_RESULTS_PATH=evaluation/results_latest.json python -m evaluation.run_final_evaluation
```

题集包含 51 道问题（42 道课程知识、9 道工具/记忆场景），覆盖来源、关键事实、工具选择与响应时间。2026-10-18 选定的收尾版本已从真实 Agent HTTP/SSE 入口完成 51 题、53 次请求，自动综合规则通过 44/51（86.3%），平均主问题耗时 12.661 秒，未触发模型降级。评分条件、未通过项及撤回实验见[收尾验证说明](docs/validation-2026-10-18.md)。该百分比不是人工语义准确率，也不代表未知问题的泛化效果。

仓库中的评测脚本直接调用 Agent Service，不经过 HTTP Router；本次 HTTP 入口验证的驱动与原始日志保存在本地。逐题报告按 `.gitignore` 不上传，仓库保留题集与运行脚本，便于在自己的知识库和环境中复测。

单元测试还覆盖显式 LangGraph 工具闭环、五个 MCP 工具注册与 stdio 调用、RAG 缓存命中、网络重试、4xx 不重试、RAG 来源与 `X-Trace-ID` 透传，以及 SSE 状态顺序。GitHub Actions 在无 API Key 的环境中自动运行这些测试。

## 调用链日志

每次 Agent 请求会向 `logs/agent_requests.jsonl` 追加一行 JSON，记录：Agent `trace_id`、同步或流式模式、实际模型、是否降级、调用工具、拦截记录、记忆是否加载、RAG 缓存命中、RAG trace、来源与总耗时。日志不记录 API Key、用户问题正文或回答正文。
