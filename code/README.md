# TraceCoder 2.1 · 智能代码修复系统

以官方 `main@cbd536eb` 与 `TraceCoder-WebDemo@49beb48` 为基线，将 TraceCoder（ICSE 2026）的方法整合为可独立运行、可部署、可接入项目与 CI 的网页系统。修复核心保持论文流程：**插桩 → 执行轨迹 → 分析与 HLLM → 修复 → 固定测试 → Rollback**。项目、网页、API 与 CLI 共用这一实现。

原目录的论文、参考脚本、数据和历史结果保持原样。新增系统代码在 `code/`，规范化数据和运行状态在 `data/`。完整交付包由 `code/scripts/package_release.py` 生成，包含代码、数据、测试、部署配置与文档，解压后无需原仓库即可运行。

官方两个分支及其完整历史保存在研究目录的 `news/`，整合未修改它们。来源与修正依据见 [官方整合说明](docs/OFFICIAL_INTEGRATION.md) 和 [来源清单](provenance/UPSTREAM.json)。`2.1.0` 是本地交付版本号，不是上游新发布版本。

## 快速启动

在包含 `code/` 和 `data/` 的目录执行：

```bash
sh code/scripts/start.sh --port 8767
```

浏览器打开 <http://127.0.0.1:8767>。需要 **macOS / Linux + Python 3.9 或以上**；Windows 可使用 WSL2 或 Docker。首次启动会创建 `code/.venv` 并安装锁定的 Flask / Waitress 及其依赖，需要访问 Python 包源；安装后无需联网或模型密钥即可运行固定演示。无需前端构建工具。若已有虚拟环境，请先运行 `code/.venv/bin/python -m pip install -r code/requirements.txt`。正式实验建议使用论文采用的 Python 3.10。

端口可通过 `--port` 修改，默认 8765。本次本机预览为 8767。模型接口依照当前要求保留后续配置，不会读取其他项目的模型密钥。

## 系统功能

| 模块 | 已实现的行为 |
| --- | --- |
| 系统总览 | 本实例实际运行数、项目数、队列占用、环境状态与最近运行；无虚构性能数字 |
| 生成与修复 | 官方需求＋测试入口，支持初始生成与冻结代码两种起点；可先保存不可变任务，模型稍后配置 |
| 官方示例 / 设置 / 帮助 | 12 个官方原始示例、浏览器预算偏好、服务器配置检查与方法说明 |
| 修复工作台 | 四个可运行案例、1,568 个基准任务、三个智能体、逐轮轨迹、诊断、代码差异、HLLM 与回滚 |
| 项目空间 | 创建项目，录入或导入任务 JSON，静态校验，保存不可变版本，查看历史版本，导出冻结输入 |
| 运行中心 | 有界队列、取消、搜索、模式与状态筛选、分页、两次记录对照、历史恢复 |
| 评审交付 | 最优源码、固定测试、候选补丁、HTML / 文本报告、全量请求响应、SHA-256 校验清单 |
| 部署中心 | 模型配置、Docker 镜像、存储、访问保护的实际检查；区分“已配置”与真实联调 |
| 工程接入 | HTTP API、CI 提交与结果回收脚本、Compose、systemd、nginx、运行状态备份 |

## 两种执行模式

| 模式 | 智能体响应来源 | 执行方式 | 可用范围 |
| --- | --- | --- | --- |
| 预设响应演示 | 明确标记的人工响应；插桩由固定 AST 程序生成 | 每次在本机隔离子进程重新执行代码与测试 | 原样四个案例，无需密钥 |
| 真实模型 | 三个角色分别调用同一个兼容接口 | 每次测试使用独立 Docker 容器 | 数据集任务、项目的冻结版本 |

预设响应不对任意新代码推理，不构成模型性能实验。历史记录查看只读取保存结果。`direct_pass` 单独表示初始实现已通过，不算作修复成功。零测试、收集失败、缺少依赖、语法错误和超时不会被判为成功。

## 四个可运行案例

| 案例 | 来源与目的 | 实际执行过程 |
| --- | --- | --- |
| 下一个星期日 | 论文 Figure 1 可观察改编；人为安排失败候选展示回滚与历史学习 | 6/7 → 0/7 → 回滚 → 7/7 |
| 账户余额中间状态 | HumanEval/3 原题及 6 个断言，人工初始缺陷 | 5/6 → 6/6 |
| 括号匹配残留栈 | ClassEval_7 原题及 15 个测试，参考实现的一处变异 | 13/15 → 15/15 |
| API 请求窗口边界 | 为应用展示编写的网关统计场景；不是生产事故或论文基准任务 | 4/6 → 6/6 |

原始预设响应、来源说明、缺陷源码和固定测试见 [data/examples](../data/examples/)。通过这些测试不代表对所有输入均正确。

```bash
python3 -B code/cli.py fixture --task paper-sunday
python3 -B code/cli.py fixture --task api-window
```

## 接入自己的项目

1. 在“项目空间”创建项目，填写来源标识；此标识仅用于追溯，不自动访问仓库。
2. 新增任务，填写需求、单文件 Python 源码与 顶层 `assert`、`unittest` 或 `check(candidate)` 测试。也可导入 [任务模板](../data/templates/api-window.task.json)。
3. 保存生成不可变的 `task_id + revision + SHA-256`。保存只解析语法和测试结构，不执行上传代码。修改时创建新版本，旧运行始终保留原输入。
4. 打开指定版本的修复工作台。服务器模型与 Docker 镜像均就绪后，可以启动真实修复。
5. 下载评审包，核对 `candidate.patch`、`best.py`、测试和报告，再由团队决定是否应用。

当前接入范围是**可提取为单文件 Python 的功能、类和业务规则**。多文件仓库、pytest、复杂构建、依赖自动安装、自动提交 PR 和自动合并不在本版本支持范围。源代码和固定测试在沙箱内共用候选命名空间；测试直接调用函数，不需要 `import candidate`。

## 后续配置模型与部署

详见 [部署手册](docs/DEPLOYMENT.md)。最小流程：

```bash
# 从包含 code/ 和 data/ 的目录构建执行镜像
# 本机需先具备可用 Docker
# 演示本身不依赖此镜像
docker build -f code/deploy/Dockerfile.sandbox -t tracecoder-sandbox:1 .

# 按 code/.env.example 在服务进程环境中设置实际配置后检查
python3 -B code/scripts/doctor.py --require-live
```

模型接口使用 `POST {TRACECODER_API_BASE}/chat/completions`，通过 `choices[0].message.content` 读取角色响应。使用官方三个核心提示词正文，解析官方分析标记和完整代码区块；同时兼容结构化 JSON 响应。三个角色共用 `TRACECODER_MODEL`，`temperature=0`。密钥仅从服务进程环境读取，网页不收集模型密钥，运行记录不保存认证头。

固定公开演示可以单独部署：

```bash
docker compose -f code/deploy/compose.yaml up --build -d
```

Compose 仅向宿主机回环地址映射 8765，关闭项目写入与实时模型执行。团队服务使用独立 Linux 主机、访问口令、HTTPS 代理与 Docker 执行器。当前版本采用单实例、单信任域共享工作空间；不宣称具备多租户隔离或高可用能力。

## CI 与 API

API 契约和请求示例见 [API.md](docs/API.md)。CI 客户端先保存冻结任务，再提交真实运行并下载评审包。口令从 CI 的环境变量 `TRACECODER_ACCESS_TOKEN` 读取，不放在命令行参数中。

```bash
python3 -B code/integrations/ci_submit.py \
  --server https://tracecoder.example \
  --project p_REPLACE_WITH_PROJECT_ID \
  --task-file data/templates/api-window.task.json \
  --output ./tracecoder-artifacts
```

将示例项目 ID 替换为项目空间显示的真实 `p-...` 标识。加 `--prepare-only` 仅验证并保存输入，适合模型配置前联调；不调用模型或执行代码。脚本不自动应用补丁，不自动重试创建请求。

## 运行状态、备份与重启

默认持久化位置：

```text
data/workspace/system.db     项目、不可变版本、可重建运行索引
data/runs/<run_id>/run.json  本次输入、轨迹、候选、诊断、反馈、角色请求响应
data/runs/<run_id>/final.py  历史最优代码（仍可能有失败测试）
```

可通过 `TRACECODER_STATE_DIR` 把运行状态迁移到独立目录；静态基准数据仍从包内 `data/` 读取。同一状态目录只允许一个服务进程。默认 2 并发、最多 6 项等待，均可配置。排队中的取消立即生效；运行中的取消在当前阶段返回后生效。重启保留结果，中断任务标记 `server_restarted`，不会自动重复付费请求。

```bash
python3 -B code/scripts/backup_state.py --output /path/to/new-backup.zip
```

备份包含 SQLite 一致性快照和原子运行文件。恢复到新的空目录后设置 `TRACECODER_STATE_DIR`，不要覆盖正在运行的工作空间。具体步骤见部署手册。

## 验证与完整交付包

```bash
sh code/test.sh
code/.venv/bin/python -B code/scripts/verify_system_http.py --output data/verification/integration-v2.1/http-system.json
python3 -B code/scripts/verify_http.py --url http://127.0.0.1:8767 --output data/verification/integration-v2.1/http-fixtures.json
python3 -B code/scripts/verify_data.py
python3 -B code/scripts/package_release.py
```

系统验收见 [SYSTEM_ACCEPTANCE.md](docs/SYSTEM_ACCEPTANCE.md)，论文对应关系见 [PAPER_MAPPING.md](docs/PAPER_MAPPING.md)，系统结构见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。旧版验证在原工作目录中单独保留，发布包只收录本版验收材料。

`code/releases/TraceCoder-2.1.0.zip` 包含运行所需的完整 `code/`、规范化 `data/`、说明与验收材料；不包含真实项目、密钥、历史运行、虚拟环境或完整旧参考脚本。解压后阅读 `START_HERE.md`，执行 `python3 code/scripts/verify_release.py .` 校验文件。数据原件校验只在完整研究目录存在时进行；可加 `--require-originals` 强制核查原论文与参考材料。

## 实现结构

```text
code/
  app.py                      Waitress 启动、访问模式、状态目录锁
  web_demo/                   官方 Flask 页面与 API 组织，Jinja 页面模板
  provenance/UPSTREAM.json    官方提交、源文件哈希与逐项迁移说明
  cli.py                      与网页共用引擎的命令行入口
  tracecoder/
    service.py                共用任务队列、取消、持久化与恢复
    official_prompts.py        来自官方 main 的三个提示词正文
    engine.py                 论文主流程与结构化运行证据
    policy.py                 Algorithm 2 的四个分支
    instrumentation.py        仅新增 print 的插桩约束
    providers.py              三个角色提示与兼容模型接口
    runner.py / worker.py     子进程、Docker 与逐项测试记录
    storage.py                SQLite 项目版本与运行索引
    reports.py                可打印报告、补丁与评审包
    readiness.py              不触发模型请求的部署检查
  static/                     九模块响应式网页，无构建依赖
  integrations/ci_submit.py   CI / 工程接入客户端
  deploy/                     Compose、隔离镜像、systemd、nginx
  scripts/                    启动、验证、备份、打包与包完整性校验
  tests/                      核心方法与系统契约自动化测试
  docs/                       方法、架构、API、部署和验收
```

论文：Jiangping Huang et al. 2026. *TraceCoder: A Trace-Driven Multi-Agent Framework for Automated Debugging of LLM-Generated Code*. ICSE ’26. DOI: [10.1145/3744916.3773187](https://doi.org/10.1145/3744916.3773187)。原论文与各数据集的作者归属和许可不因整理改变，见 [NOTICE.md](NOTICE.md)。
