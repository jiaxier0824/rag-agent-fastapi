# RAG Agent FastAPI

一个基于 FastAPI 的课程资料智能问答 Agent。它把已有 RAG 服务封装为可调用工具，并支持学习计划、Redis 缓存、多模型降级与 SSE 流式输出。

## 核心能力

- 通过 HTTP 调用独立部署的 RAG 服务检索课程资料
- Agent 自动选择工具：
  - `search_course_knowledge`：检索课程资料
  - `create_study_plan`：生成并保存学习计划
  - `get_current_study_plan`：读取已有学习计划
- Redis 问答缓存与学习计划持久化
- `qwen3-max` 主模型，`qwen-plus` 失败降级
- SSE 流式响应，实时返回“正在检索资料”等状态
- RAG 请求重试、超时处理与调用链 `trace_id`
- Docker Compose 一键启动 MySQL、Redis、RAG、Agent

## 技术栈

Python、FastAPI、LangChain、通义千问、Redis、MySQL、Chroma、Docker、Docker Compose。

## 架构

前端/调用方 → FastAPI Agent → LLM 选择工具 → RAG 服务或学习计划工具 → Redis/MySQL/Chroma。

RAG 服务是独立项目，Agent 通过 `RAG_API_BASE_URL` 调用它，而不是直接耦合 RAG 源码。

## 本地配置

复制示例配置：

```bash
cp .env.example .env
```

然后在 `.env` 中填写自己的：

```env
DASHSCOPE_API_KEY=你的百炼API密钥
```

不要上传 `.env`。

## Docker 启动

本项目的 `docker-compose.yml` 会同时启动 MySQL、Redis、RAG、Agent。

```bash
docker compose up --build -d
```

查看运行状态：

```bash
docker compose ps
```

接口文档：

- RAG：http://127.0.0.1:8000/docs
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

已完成 5 道课程问题的端到端测评：

- 关键事实正确率：80%
- 工具选择正确率：100%
- 平均响应时间：13.176 秒

## 后续迭代

- 优化 RAG 召回与重排序，解决截止日期类问题的漏检索
- 增加用户记忆与更多外部工具
- 增加 Web 前端和可视化调用链