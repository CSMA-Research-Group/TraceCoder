# 数据与执行证据

本目录由根目录随附数据生成，原材料保持原样。正常启动无需联网下载或重新转换。

| 数据集 | 条数 | 本目录入口 | 原来源 |
| --- | --- | --- | --- |
| HumanEval | 164 | `benchmarks/humaneval.jsonl` | `../datasets/human_eval/data/HumanEval.jsonl.gz` |
| HumanEval+ | 164 | `benchmarks/humanevalplus.jsonl` | `../datasets/human_eval_plus/data/test-00000-of-00001-5973903632b82d40.parquet` |
| ClassEval | 100 | `benchmarks/classeval.jsonl` | `../datasets/ClassEval/data/ClassEval_data.json` |
| BigCodeBench | 1140 | `benchmarks/bigcodebench.jsonl` | `../datasets/BigCodeBench/data/v0.1.4-00000-of-00001.parquet` |

`source_manifest.json` 保存源文件及转换输出的 SHA-256、条数和转换说明。仅做序列化格式转换，保留原字段、原测试及原参考解；模型任务读取层主动排除参考解。BigCodeBench 网页默认使用 Complete 题面，Instruct 原字段仍保存在 JSONL 中，但未另设 Instruct 模式或进行两种口径的完整对比实验。

`examples/` 的四个演示案例明确说明缺陷和响应为人工编写。HumanEval/3 和 ClassEval_7 使用原测试，星期日案例按论文 Figure 1 补全七天测试。它们不是新采集的模型失败数据。

`runs/<run_id>/run.json` 是一次执行的完整证据，`final.py` 是历史最优实现。记录只在实际启动时生成；历史浏览不会重新运行。该目录默认被 Git 忽略，便于保留本机执行记录而不意外提交任务内容。

`verification/` 保存 HTTP 验证、执行器兼容性验证、浏览器截图和验收摘要；当前边界见 [系统验证报告](../code/docs/SYSTEM_ACCEPTANCE.md)。`original_materials.sha256.json` 是新建目录时对 34 个原始材料文件保存的校验清单。旧版验收只保留在原工作目录，发布包包含本版验收。

若需从原材料重建 JSONL，在有 PyArrow 的 Python 环境中运行：

```bash
python3 code/scripts/prepare_data.py
```

该脚本不会修改根目录原数据或已有运行记录，会重建本目录的数据转换和教学案例。不要把后续修改过的教学案例未经保存就重建。

数据及参考实现的原有来源、作者归属和许可保持不变。本次只记录本地材料来源与哈希，没有重新授权数据或声称重新验证了全部数据集的上游版本。

## 系统 2.0 的数据分层

新增 API 请求窗口案例是为系统接入展示编写的业务场景，并非生产事故或论文基准。

- `templates/`：项目导入模板；只包含需求、初始代码与测试，不含预设修复答案。
- `workspace/system.db`：首次启动创建的项目、不可变任务版本与运行摘要索引；不进入发布 ZIP。
- `runs/`：本实例逐次执行证据；不进入发布 ZIP。
- `verification/system-v2/`：本次系统级测试、浏览器截图与固定演示验收记录；属于验证证据，不是模型性能实验。

设置 `TRACECODER_STATE_DIR` 后，`workspace/` 与 `runs/` 写入该目录，规范化基准与演示模板继续从包内读取。发布包包含全部 1,568 条规范化基准任务；不包含用户项目、模型密钥或旧实验 CSV。

## 官方整合 2.1

`official/examples.json` 原样保存官方 WebDemo `49beb48` 中 12 个示例对象（名称、描述、测试、分类、难度、提示）；来源文件哈希见 `code/provenance/UPSTREAM.json`。新增生成任务的 `start_mode=generate`，冻结需求与测试，初始代码为空。旧任务版本继续按 frozen 解释。

本版验收在 `verification/integration-v2.1/`。`backups/pre-official-integration-*/` 保留整合前代码与数据 ZIP 及完整哈希清单，不进入发布包。研究目录中的 `news/` 保持官方检出原样，发布包无需该目录即可运行。
