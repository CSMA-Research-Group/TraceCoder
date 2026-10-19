# 作者归属与材料范围

本版以 CSMA-Research-Group/TraceCoder 官方 main 与 TraceCoder-WebDemo 为基线整合。三个核心提示词正文和 12 个示例来自官方源码；Flask 页面组织延续官方结构，执行、状态管理与界面按论文和部署要求改造。完整提交、文件哈希与迁移说明见 provenance/UPSTREAM.json。没有在新服务中直接导入旧全局模型连接模块。

论文：Jiangping Huang, Wenguang Ye, Weisong Sun, Jian Zhang, Mingyue Zhang, Yang Liu. 2026. *TraceCoder: A Trace-Driven Multi-Agent Framework for Automated Debugging of LLM-Generated Code*. ICSE ’26. DOI: 10.1145/3744916.3773187。

规范化基准数据来自用户工作目录内提供的 HumanEval、HumanEval+、ClassEval、BigCodeBench 材料。保留原字段、题目 ID、测试和来源文件 SHA-256，详见 `data/source_manifest.json`。转换不改变各数据集及参考实现的原作者归属或原许可，也不额外授予第三方材料的再分发权。

发布 ZIP 供本项目交付与部署使用，不包含原论文 PDF、根目录参考代码或旧实验结果。若向第三方公开发布完整数据包，应按原始数据集许可准备对应通知与归属文件；当前目录没有统一的独立许可文件，因此未推定全部材料适用同一开源许可证。

四个演示案例均明确标注响应与缺陷来源；新增 API 窗口案例和项目任务模板为说明应用工作流编写，不是采集到的生产缺陷或真实模型输出。

官方仓库归属 CSMA Research Group；两个已拉取分支中未发现 LICENSE 文件，本包未另行声明上游源码适用其他许可证。Flask、Waitress 及安装依赖保留各自发行包内的许可与归属。
