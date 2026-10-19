# HTTP API 与工程接入

Base URL 为实例根地址。响应默认为 JSON。保护实例需在每次 `/api/` 请求中设置 `X-TraceCoder-Token`；不要把口令放入 URL。写入请求使用 `Content-Type: application/json`，请求体最多 250 KB。跨来源网页写请求被拒绝。接口不读取或返回模型密钥。

## 项目与不可变任务

```http
POST /api/projects
{"name":"API 网关","description":"窗口统计回归","source_ref":"repo@commit"}
```

返回 HTTP 201，包含 `id`（`p-...`）、名称、说明、来源标识和 UTC 创建时间。`GET /api/projects` 列出项目；`GET /api/projects/{project_id}` 返回项目及各任务的最新版本。

```http
POST /api/projects/{project_id}/tasks
{
  "title": "窗口计数",
  "description": "统计 (now-window, now] 内的请求，排除过期与未来请求。",
  "initial_code": "def count_recent(timestamps, now, window):\n    return sum(now-window <= t <= now for t in timestamps)\n",
  "test": "def check(candidate):\n    assert candidate([40,41,100],100,60) == 2\n    assert candidate([],100,60) == 0\n",
  "entry_point": "count_recent",
  "source_ref": "repository@commit:window.py"
}
```

返回 HTTP 201，包含 `id`（`t-...`）、`project_id`、`revision`、`revision_sha256` 和冻结内容。任务标题最多 120 字符，需求 16,000，代码/测试各 60,000，来源标识 300。`check(candidate)` 必须有入口名和断言；unittest 使用 `test...` 测试方法。只做静态校验，不在此接口执行代码。

向 `POST /api/projects/{project_id}/tasks/{task_id}` 提交相同字段创建下一版本。使用 SQLite 写事务分配版本号，并发请求也不覆盖已有版本。

- `GET /api/project-tasks/{task_id}`：最新版本。
- `GET /api/project-tasks/{task_id}?revision=1`：指定版本。
- `GET /api/project-tasks/{task_id}/revisions`：版本列表、标题、哈希与时间。
- `GET /api/templates`：可以原样导入的任务模板。

## 运行提交

项目任务必须指定固定版本，不能在提交运行时覆盖源码或测试：

```http
POST /api/runs
{"mode":"live","dataset":"project","task_id":"t-...","revision":1,"max_rounds":5,"patience":2,"timeout":10}
```

返回 HTTP 202 和运行快照。创建请求不自动重试，避免产生重复付费运行。超过并发与等待容量会返回错误，已入队任务持久化为 `queued`。

预设演示请求：

```http
POST /api/runs
{"mode":"fixture","dataset":"demo","task_id":"api-window"}
```

数据集生成请求：

```http
POST /api/runs
{"mode":"live","dataset":"humaneval","task_id":"HumanEval/3","start_mode":"generate"}
```

已有数据集失败代码可以使用 `start_mode=frozen` 和 `initial_code`。项目模式固定使用保存版本，不支持初始生成。预设模式禁止提交自定义代码、测试或生成请求。

## 查询、取消与评审

| 方法与路径 | 返回 |
| --- | --- |
| `GET /healthz` | 无需口令的存活检查 `{ "ok": true }` |
| `GET /api/config` / `/api/deployment` | 版本、配置是否存在、镜像检查、模式能力；无模型调用 |
| `GET /api/overview` | 本实例实际计数、执行容量和最近运行 |
| `GET /api/tasks` | 四个演示案例与四套基准任务索引；不返回修复预设步骤 |
| `GET /api/task?dataset=humaneval&id=HumanEval%2F3` | 题目与固定测试；不返回参考解 |
| `GET /api/runs` | 最近 50 项运行摘要，保留 v1 兼容接口 |
| `GET /api/runs/search?query=API&mode=fixture&status=completed&limit=15&offset=0` | `{items,total,limit,offset}`，支持 `project_id`，最大 limit=100 |
| `GET /api/runs/{run_id}` | 完整快照：输入、逐轮轨迹、调用、HLLM、决策、哈希、停止原因 |
| `POST /api/runs/{run_id}/cancel`，body `{}` | 取消排队或标记在当前阶段完成后停止；已结束返回 `requested=false` |
| `GET /api/runs/{run_id}/export` | 原始 JSON 下载 |
| `GET /api/runs/{run_id}/report` | 转义项目与模型内容的独立 HTML 报告，可打印 |
| `GET /api/runs/{run_id}/bundle` | ZIP 评审包 |
| `GET /api/release` | 发布包版本、大小、SHA-256 或尚未构建说明 |
| `GET /api/release/download` | 已构建的完整代码 ZIP |

评审包包含 `run.json`、`initial.py`、`best.py`、`fixed_tests.py`、`candidate.patch`、`report.html`、`report.txt`、`README.txt` 与 `SHA256SUMS`。补丁目标名是评审占位路径 `candidate.py`，应用前必须核对原工程真实路径。

`GET /api/runs/{id}` 在运行中可轮询；建议 0.5–2 秒间隔。数据量随测试与模型输出增加，列表接口仅返回摘要。全部时间为 ISO 8601 UTC；前端按浏览器时区显示。

## 错误与边界

- 400：请求字段/类型/输入/任务预算无效，模型或镜像尚未就绪、队列已满、提交频率超限。
- 401：缺失或错误的实例访问口令。
- 403：Host、跨来源、公开演示写入保护拒绝。
- 404：不存在的项目、版本或运行；公开模式访问私有记录也返回 404。
- 413：请求体大小无效。415：写请求不是 JSON。
- 500：服务端存储或启动失败，响应不返回内部堆栈。

公开演示不会提供项目数据或真实模型运行记录。此 API 的授权单位为整个实例；可信团队中拥有口令的人可查看该实例全部项目和运行，不是按用户隔离。

## CI 客户端

[ci_submit.py](../integrations/ci_submit.py) 是仅依赖标准库的真实客户端。它保存任务、固定版本、提交运行、等待结束、取回 JSON 与评审包，不应用补丁。

- `--prepare-only`：只保存版本，返回 `model_called=false`、`code_executed=false`，适合无模型环境联调。
- 默认等待 900 秒，超过后请求取消；服务端仍保留证据，可在运行中心查看。
- 退出码 0：全部测试通过，或输入准备完成；1：运行未全部通过/等待超时；2：客户端参数错误；3：HTTP/传输失败。
- 客户端拒绝远程明文 HTTP、含凭据 URL 和重定向。口令仅从环境变量读取。

示例变量和运行命令见 [主说明](../README.md)。`verify_system_http.py` 已覆盖真实客户端的 prepare-only 通路；真实模型的 CI 修复仍需后续接口配置后验收。

## 官方 WebDemo 兼容入口（2.1）

保留官方路径，统一进入同一持久化作业服务。协议变更：`POST /api/run_tracecoder` 现在返回 **202 + Run 对象**，不是同步 `steps`。旧客户端需使用返回的 `id` 查询 `/api/runs/{id}`。不再接受 `api_key/base_url/model_name`，出现这些字段返回 400；模型配置由服务器管理。

| 接口 | 行为 |
| --- | --- |
| `GET /api/examples[/{id}]` | 官方 WebDemo 12 个原始示例对象 |
| `GET /api/provenance` | 固定上游提交、源文件哈希、迁移说明 |
| `GET /api/status` | 当前实例配置状态 |
| `GET /api/models` | 服务器已配置模型名称；未配置返回空数组 |
| `POST /api/validate_config` | 空 JSON 对象，读取配置并返回 `model_called=false`，不发起请求 |
| `POST /api/generation/tasks` | 保存生成/修复输入版本，201；无需配置模型 |
| `POST /api/run_tracecoder` | 提交固定版本或官方需求＋测试，202 |
| `POST /api/run_tracecoder_stream` | 同一提交逻辑，SSE 状态事件；仍使用同一鉴权 |
| `POST /api/stop_tracecoder` | `{ "run_id": "..." }`，只停止指定运行 |

保存输入示例：

```json
{
  "task": {
    "title": "两倍计算",
    "description": "实现 double(x)，返回整数 x 的两倍。",
    "start_mode": "generate",
    "initial_code": "",
    "test": "assert double(0)==0\nassert double(3)==6",
    "entry_point": "",
    "source_ref": "team-repo@commit / ISSUE-42"
  }
}
```

可选 `project_id` 指定已有项目，省略则进入“生成与修复任务”项目；修改已有任务传入同一 `project_id` 和 `task_id`，保存为新 revision。`start_mode` 默认 `frozen`，必须提供初始代码；`generate` 必须留空初始代码。测试支持顶层 assert、unittest 或 check(candidate)，后者需 entry_point。已有历史版本未包含 start_mode 时按 frozen 解释。

运行指定版本示例（模型与 Docker 就绪后）：

```json
{
  "mode": "live", "task_id": "t-REPLACE", "revision": 1,
  "max_debug_attempts": 5, "max_no_improvement": 2,
  "timeout": 10, "enable_instrumentation": true
}
```

参数对应 `Config.max_rounds/patience/timeout/enable_instrumentation`。数字必须为整数，轮数及停滞为 1–5，执行超时为 1–30 秒。插桩开关必须是 JSON 布尔值，关闭会明确标为消融配置。标准 `/api/runs` 同样支持 `enable_instrumentation`。

直接需求提交使用 `problem_description`、`test_cases`、可选 title/entry_point/source_ref/initial_code/start_mode；就绪后自动保存输入版本再进入队列。若已传 task_id，不允许再覆盖内容，应先保存新版本。

无密钥演示可提交 `{ "mode":"fixture", "task_id":"paper-sunday" }`；不能附带自定义代码或测试。

SSE 每条消息是 `event: 类型` 和 `data: JSON`：`run` 给出运行 ID；`step` 为递增序号事件；`result` 为完整终态 Run（包括失败或取消）；`resume` 指示连接达到 25 秒限制，任务继续，客户端按其 url 轮询。不要因断流重新提交任务。最多 4 条流式连接，额外连接返回 429。取消等待当前模型或测试阶段结束。
