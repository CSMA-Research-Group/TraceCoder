# TraceCoder 2.1 网页系统

系统实现位于 [code/](code/README.md)，配套数据位于 [data/](data/README.md)。本版结合官方 `main` 与 `TraceCoder-WebDemo` 的内容，按论文实现插桩、轨迹分析、历史学习、修复与回滚。根目录原有研究代码和 README 保持原样。

在仓库根目录启动：

```bash
sh code/scripts/start.sh --port 8767
```

浏览器打开 <http://127.0.0.1:8767>。需要 macOS / Linux 与 Python 3.9 或以上；Windows 可使用 WSL2。首次启动会安装锁定依赖，随后四个固定演示无需模型密钥即可运行。演示中的智能体响应由人工编写，代码与测试每次实际执行。

- [完整独立代码包](code/releases/TraceCoder-2.1.0.zip) · [SHA-256 校验值](code/releases/TraceCoder-2.1.0.zip.sha256)
- [系统使用说明](code/README.md) · [部署手册](code/docs/DEPLOYMENT.md) · [API](code/docs/API.md)
- [论文对应关系](code/docs/PAPER_MAPPING.md) · [官方整合说明](code/docs/OFFICIAL_INTEGRATION.md) · [验收记录](code/docs/SYSTEM_ACCEPTANCE.md)

2.1.0 标识本次系统交付版本。58 项自动化测试与独立代码包验收已通过；这些结果不代表论文模型性能指标复现。真实模型接口、Docker 执行和公网部署尚未联调，按后续配置阶段推进。

Git 中提供代码、必要数据、部署配置与当前验收材料。本地参考检出 `news/`、原论文 PDF、运行记录、项目状态、密钥和备份保留在研究工作目录中，不随本次提交上传。
