# 部署与运维

本系统提供两种部署配置。代码包具备完整的本地运行能力；是否完成模型联调、镜像构建和公网部署，以目标主机的实际验收为准。

## A. 本地工作空间

```bash
sh code/scripts/start.sh --port 8767
```

仅监听 `127.0.0.1`。预设案例不需要任何外部服务；项目导入、版本、历史查询、报告和打包均可独立使用。准备好模型后，按 B 的隔离镜像步骤配置。

应用由 Flask + Waitress 提供服务，版本锁定在 `requirements.txt`。首次启动自动建立虚拟环境并安装依赖，需要联网；后续固定演示可离线使用。升级已有环境时执行 `code/.venv/bin/python -m pip install -r code/requirements.txt`。`requirements-data.txt` 只用于从原始 Parquet 重建规范化数据，发布包已包含结果，通常不需要安装。

## B. 可信团队的真实模型工作台

建议独立 Linux 主机，Python 3.10、Docker、HTTPS 反向代理。应用服务账户拥有 Docker 权限；该权限具有主机管理能力，因此主机应为本服务专用。候选容器不继承服务密钥，不挂载项目目录或 Docker socket。

1. 将交付包解压到 `/opt/TraceCoder`，确保此目录下直接包含 `code/` 和 `data/`。校验 `MANIFEST.json`。
2. 创建专用 `tracecoder` 账户，按主机现有管理方式授予 Docker 使用权限。
3. 从包根目录安装应用依赖并构建隔离镜像：

```bash
python3 -m venv code/.venv
code/.venv/bin/python -m pip install -r code/requirements.txt
docker build -f code/deploy/Dockerfile.sandbox -t tracecoder-sandbox:1 .
```

4. 将 [环境模板](../.env.example) 的变量写入 `/etc/tracecoder.env`，由服务账户/系统管理设置严格权限。文件内采用 `NAME=value` 形式，不包含 `export`。至少设置：

```text
TRACECODER_API_BASE=https://your-provider.example/v1
TRACECODER_MODEL=your-model-name
TRACECODER_API_KEY=your-provider-key
TRACECODER_ACCESS_TOKEN=a-long-random-instance-token
TRACECODER_SANDBOX_IMAGE=tracecoder-sandbox:1
TRACECODER_WORKERS=2
TRACECODER_QUEUE_LIMIT=6
```

`TRACECODER_API_BASE` 不含 `/chat/completions`。远程接口必须为 HTTPS；本机模型允许 HTTP。API key 可为空，适用于不要求密钥的内部兼容服务。应用不会自动解析 `.env` 文件，也不会消费机器上其他命名的密钥。

5. 按主机实际路径安装 [systemd 模板](../deploy/tracecoder.service.example)，再执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tracecoder
sudo systemctl status tracecoder
```

systemd 通过 `StateDirectory=tracecoder` 创建 `/var/lib/tracecoder`，运行状态与只读程序文件分离。停止等待上限 120 秒，覆盖默认模型请求超时；修改 API 超时时相应调整该值。

6. 在已有 HTTPS 虚拟主机中配置 [nginx 片段](../deploy/nginx.conf.example)，将外部请求转发到 `127.0.0.1:8765`。限制服务器入站端口，不直接将未加 TLS 的服务端口作为公网入口。浏览器点击右上角 TC 按钮，输入实例访问口令。口令只保留在当前页面内存，刷新后需重新输入。
7. 以相同服务环境运行 `doctor.py --require-live`，或查看网页“部署中心”。检查通过只证明配置可读和镜像存在；仍应执行一个自有小任务，验收真实请求、容器退出、失败日志和评审包后，再接入实际业务。

执行容器使用 `--network=none`、只读根文件系统、非 root 用户、去除 capabilities、`no-new-privileges`、CPU/内存/PID 上限和临时文件空间。无网络可能导致依赖外部服务的业务测试无法运行，应先抽取离线逻辑或提供固定测试替身。

基础镜像含标准库及 NumPy 1.26.4。部分 ClassEval、BigCodeBench 任务需要额外依赖，应构建固定版本的专用镜像，并指定 `TRACECODER_SANDBOX_IMAGE`。本系统不会根据提交代码自动安装依赖。

## C. 固定公开演示

```bash
docker compose -f code/deploy/compose.yaml up --build -d
docker compose -f code/deploy/compose.yaml logs --tail=100
```

- 服务采用 `--public-demo`，只接受四个原样演示案例。项目写入、项目读取与真实模型运行关闭，既有私有运行也不向公开接口暴露。
- 宿主机端口为 `127.0.0.1:8765`，由 HTTPS 代理公开；容器进程为非 root，根文件系统只读，运行状态写入命名卷 `state`。
- 无需模型密钥、Docker socket 或执行镜像。不要复用私有工作台的状态卷作为公开演示卷。
- 容器镜像默认不附带发布 ZIP；网页代码包入口需要先执行打包脚本生成 `code/releases/`。本地交付目录已提供生成后的 ZIP。

## 配置表

| 环境变量 | 默认值 / 行为 |
| --- | --- |
| `TRACECODER_STATE_DIR` | 包根目录 `data`；运行状态放在其 `workspace/`、`runs/` |
| `TRACECODER_WORKERS` | 2；允许 1–8 |
| `TRACECODER_QUEUE_LIMIT` | 6；允许 0–32，超额返回明确错误 |
| `TRACECODER_API_BASE` / `TRACECODER_MODEL` | 默认未配置 |
| `TRACECODER_API_KEY` | 默认空，仅服务端读取 |
| `TRACECODER_API_TIMEOUT` | 90 秒 |
| `TRACECODER_MAX_OUTPUT_TOKENS` | 4096 |
| `TRACECODER_SANDBOX_IMAGE` | `tracecoder-sandbox:1` |
| `TRACECODER_ACCESS_TOKEN` | 默认空；非本机监听的私有模式强制要求 |
| `TRACECODER_PYTHON` | 启动/测试脚本默认使用 `code/.venv/bin/python`；可指定已有环境的绝对路径 |

任务默认最多 5 轮、连续无提升上限 2、单次执行 10 秒；网页可调整但不允许超过论文的 5 轮上限。每个来源 IP 每分钟最多提交 12 次运行。应用不信任任意 `X-Forwarded-For`；反向代理后的用户共享此应用级限额，可在 nginx 层另设每客户端限制。

## HTTP 服务组织

`app.py` 使用 Waitress 的 12 个 HTTP 工作线程、1 个进程；修复队列独立，默认 2 并发。不得给同一状态目录启动多个 WSGI 进程。直接执行启动脚本可保留状态目录独占锁和有序退出；不使用 Flask debug/reloader。流式接口最多同时 4 条连接，每条 25 秒后可释放 HTTP 线程，返回 `resume` 事件；任务继续执行，客户端使用运行查询接口读取后续状态。断开流不会取消任务，取消必须明确指定 run_id。

## 状态与退出

- `queued` → `running` → `completed / error / cancelled`。
- `completed` 不是“全部测试通过”的同义词；还要查看 `stop_reason` 和 `final_eval.passed`。
- `all_tests_passed`：修复后的候选通过固定测试；`direct_pass`：初始代码就已通过。
- `patience_exhausted / round_budget_exhausted`：按预算正常停止，最优代码可能仍失败。
- `server_restarted`：服务发现上次未结束的记录，保存中断状态，不自动续跑或重发模型请求。
- 排队取消不调用模型；运行中取消等待当前请求或执行结束。停止不会删除历史证据。

## 备份、恢复、升级

```bash
# 不覆盖已有备份；默认备份 TRACECODER_STATE_DIR
python3 -B code/scripts/backup_state.py --output /backups/tracecoder-new.zip
```

备份包含 SQLite 一致性快照及逐个原子读取的运行证据。运行索引可在服务启动时由 `run.json` 重建；正在运行的任务会被备份为当时状态。项目源码和模型交互可能属于项目私有资料，备份应遵循项目访问范围。

恢复时先停止旧服务，**解压到新的空目录**，检查 `backup-manifest.json` 的文件哈希，再将 `TRACECODER_STATE_DIR` 指向该目录并启动。不要覆盖运行中的目录；保留旧目录作为回退。模型与访问口令不在备份内，继续由部署环境管理。

升级时先备份状态、停止服务，再将新代码包解压到新的程序目录；复用原状态目录即可。本版本数据库 schema 为 1，后续结构变更必须提供迁移。运行文件保存本次核心模块哈希，旧记录不会被标记为新版本重跑结果。

## 当前适用范围

一个服务进程、一套共享项目空间、同一信任域的小团队。访问口令保护实例，但没有用户角色、多租户记录隔离、分布式执行节点或高可用调度。高风险业务补丁必须在原工程环境中复核。系统输出的是候选补丁与证据包，不自动部署代码。

本次实际验收的项目、队列、重启恢复、访问控制、CI 输入准备与本地浏览器记录见 [SYSTEM_ACCEPTANCE.md](SYSTEM_ACCEPTANCE.md)。Docker 镜像构建与真实模型联调须在目标部署主机完成。
