# RAG Agent FastAPI

一个基于 FastAPI 的课程资料智能问答 Agent。它把独立 RAG 服务封装为可调用工具，并支持可控记忆、多课程学习计划、工具保护、多模型降级与 SSE 流式输出。

## 核心能力

- 通过 HTTP 调用独立部署的 RAG 服务检索课程资料
- Agent 自动选择工具：
  - `search_course_knowledge`：检索课程资料
  - `create_or_update_study_plan`：按课程创建或更新学习计划
  - `get_study_plan` / `list_study_plans`：读取指定课程计划或计划列表
  - `save_learning_preferences`：仅保存用户明确表达的稳定学习偏好
- Redis 入口问答缓存、短期会话记忆、显式偏好与多课程计划持久化；相同问题命中缓存时跳过 Agent 与 RAG 调用
- 请求级工具保护：参数校验、重复工具调用拦截、最大调用次数限制、结构化成功/失败结果
- `qwen-turbo` 主模型，`qwen-plus` 失败降级
- SSE 流式响应，实时返回“正在检索资料”等状态
- RAG 请求重试、超时处理与调用链 `trace_id`
- 透传 RAG V2 的来源文件与调用链，最终回答可追溯资料依据
- Agent JSONL 调用日志：模型切换、工具/拦截记录、记忆加载、缓存、来源、耗时与 RAG 调用链
- Docker Compose 启动 Agent 与 Redis；Agent 通过 HTTP 调用独立运行的 RAG 服务

## 技术栈

Python、FastAPI、LangChain、通义千问、Redis、Docker、Docker Compose。

## 架构

前端/调用方 → FastAPI Agent → LLM 选择工具 → RAG 服务或学习计划工具 → Redis。

RAG 服务是独立项目 `RAG_FastAPI`，Agent 通过 `RAG_API_BASE_URL` 调用它，而不是直接耦合 RAG 源码。Agent 将自己的 `X-Trace-ID` 传给 RAG；RAG 返回的 `sources` 会由 Agent 透传给调用方。

## 本地配置

复制示例配置：

```bash
cp .env.example .env
```

然后在 `.env` 中填写自己的：

```env
DASHSCOPE_API_KEY=你的百炼API密钥
AGENT_MODEL_NAME=qwen-turbo
AGENT_FALLBACK_MODEL_NAME=qwen-plus
AGENT_SHORT_MEMORY_MAX_TURNS=4
AGENT_MAX_TOOL_CALLS=6
```

不要上传 `.env`。应填写归属项目业务空间的 API Key；不要使用聊天记录中已经暴露过的旧 Key。

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

运行真实测评：

```bash
python -m evaluation.run_evaluation
```

课程资料端到端测评覆盖来源命中、响应时延与缓存效果；V3 还通过离线单测覆盖多课程隔离、显式偏好记忆、重复工具调用保护和缓存短路。脱敏的真实集成测试摘要见 [docs/INTEGRATION_TEST_RESULTS.md](docs/INTEGRATION_TEST_RESULTS.md)。

最终配置下，15 道跨课程首次问答的 HTTP 成功率为 100%，来源命中率为 93.33%，平均响应时间为 5.73 秒；同一问题、同一会话的第二次请求由 Redis 入口缓存直接返回，端到端耗时为 44.6 毫秒。

单元测试还覆盖 RAG 缓存命中、网络重试、4xx 不重试、RAG 来源与 `X-Trace-ID` 透传，以及 SSE 状态顺序。GitHub Actions 在无 API Key 的环境中自动运行这些测试。

## 调用链日志

每次 Agent 请求会向 `logs/agent_requests.jsonl` 追加一行 JSON，记录：Agent `trace_id`、同步或流式模式、实际模型、是否降级、调用工具、拦截记录、记忆是否加载、RAG 缓存命中、RAG trace、来源与总耗时。日志不记录 API Key、用户问题正文或回答正文。

## 后续迭代

- 通过人工标注用例继续扩展工具选择、两步工具链和记忆命中测评
- 增加鉴权、限流与多用户隔离
- 视业务需要增加异步队列与可视化运维面板
