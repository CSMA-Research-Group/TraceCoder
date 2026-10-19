# TraceCoder 2.1 官方整合验收

验收日期：2026-10-19（Asia/Shanghai）。环境：macOS / Python 3.9.6 / Flask 3.1.3 / Waitress 3.0.2 / Chrome。核心按论文 Algorithm 1/2；官方代码与本地工程改造边界见 [OFFICIAL_INTEGRATION.md](OFFICIAL_INTEGRATION.md)。

| 验收项 | 实际结果 | 证据 |
| --- | --- | --- |
| 保留官方基线 | main 60 个、WebDemo 101 个跟踪文件未变，两树干净；原论文与材料 34 个文件哈希不变 | [upstream-preservation.json](../../data/verification/integration-v2.1/upstream-preservation.json) |
| 官方内容保留 | 三个角色提示词正文一致；12 个示例对象一致 | 同上 |
| 核心与系统契约 | 58 项测试通过；含回滚四分支、完整历史、超时/跳过/零测试、assert 适配、实际参数、SSE、生成版本、队列与持久化 | [unit-tests.txt](../../data/verification/integration-v2.1/unit-tests.txt) |
| 实际 HTTP | 79 次检查通过；访问保护、项目版本、运行/报告、重启恢复、公开模式隔离、CI prepare-only | [http-system.json](../../data/verification/integration-v2.1/http-system.json) |
| 数据与执行器 | 4 套 1,568 条数据哈希通过；164 / 164 原 HumanEval 参考解通过测试 | [data_and_evaluator.json](../../data/verification/integration-v2.1/data_and_evaluator.json) |
| 浏览器 | 官方示例载入和版本保存、生成任务编辑、完整回滚案例、历史深链接刷新、九页手机布局；无捕获的控制台错误 | [browser-checks.json](../../data/verification/integration-v2.1/browser-checks.json) |
| 独立交付包 | 空目录解压、单独虚拟环境安装、HTTP 与 CLI 执行、完整文件校验；使用相同运行代码 | [clean-package.json](../../data/verification/integration-v2.1/clean-package.json) |

手机验收宽度 390 px、高度 844 px，页面无整体横向溢出；窄屏导航和源代码区域保留局部滚动。截图：[系统总览](../../data/verification/integration-v2.1/overview-desktop.jpg)、[生成任务](../../data/verification/integration-v2.1/generate-desktop.jpg)、[修复工作台](../../data/verification/integration-v2.1/workbench-desktop.jpg)、[手机工作台](../../data/verification/integration-v2.1/workbench-mobile.jpg)。

## 四个固定案例的真实测试执行

| 案例 | 初始 | 候选过程 | 最优 | 修复轮次 |
| --- | --- | --- | --- | --- |
| 星期日 / 论文 Figure 1 改编 | 6/7 | 第一轮 0/7 回滚；第二轮使用完整失败记录 | 7/7 | 2 |
| HumanEval/3 中间余额 | 5/6 | 修复循环中的余额检查 | 6/6 | 1 |
| ClassEval_7 括号残留栈 | 13/15 | 修复结束时的残留栈判定 | 15/15 | 1 |
| API 请求窗口 | 4/6 | 修复窗口左边界条件 | 6/6 | 1 |

智能体响应均为人工编写，代码与测试每次执行。完整轨迹见 [snapshots](../../data/verification/integration-v2.1/snapshots/)。164 个参考解检查是执行器兼容性验证，不是模型生成或修复实验。

官方入口参数测试将星期日案例设为 1 轮、停滞 4、执行超时 5 秒、关闭插桩，确认保存的 Config 与实际调用一致，结果以预算耗尽结束、最优仍为 6/7，没有被显示为全部通过。

生成流程的测试使用明确标记的脚本响应替身和本机可信测试执行器，确认“生成 → 失败测试 → 修复”的服务调用链与保存版本一致，且测试内容不进入初始生成请求。这不证明外部模型或 Docker 已联调。

浏览器准备的“官方示例 · 两数之和（输入准备）”只是持久化输入，未调用模型。历史项目及旧运行保持可访问；旧记录不会变成新版本实测。

## 当前未执行的阶段

- 按要求未配置、未调用外部模型接口。
- 本机没有 Docker，未实际构建部署镜像、验收容器隔离或部署公网服务；Compose、systemd、nginx 和执行镜像配置已随包提供。
- 未复现论文全量性能指标，也没有真实业务效果、费用或延迟结论。
- 当前支持单文件 Python 与一种固定测试入口；不自动修改多文件仓库，不提供多租户隔离或高可用调度。

后续在目标主机配置模型和 Docker，以自有小任务完成真实联调，再接入团队的人工补丁评审流程。
