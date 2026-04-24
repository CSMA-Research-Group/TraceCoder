# TraceCoder

**TraceCoder** 是一个基于大语言模型的代码生成与自动调试系统。该系统采用轨迹驱动的调试方法，通过代码插桩、错误分析和迭代修复三个步骤，实现对生成代码的自动调试和优化。

## 项目简介

TraceCoder 系统旨在帮助开发者快速生成高质量的代码，并自动调试和修复代码错误，提高软件开发的效率和质量。系统支持多种大语言模型（如 Gemini、DeepSeek、Qwen 等），并提供 Web 图形界面和命令行两种使用方式。

### 核心特性

- **智能代码生成**：基于精心设计的提示词工程，引导大语言模型生成高质量代码
- **自动错误修复**：采用三步式调试机制（代码插桩 → 错误分析 → 代码修复）
- **轨迹驱动调试**：通过添加打印语句追踪变量状态和执行流程
- **多智能体协作**：支持分析器、规划器和实现器三个智能体的协作
- **历史学习机制**：利用历史失败记录避免重复错误
- **智能回滚机制**：当连续多次修复无改善时自动回滚到最佳版本

### 支持的数据集

| 数据集 | 描述 | 格式 |
|--------|------|------|
| HumanEval | 经典的 Python 函数生成基准测试 | JSONL |
| HumanEval+ | HumanEval 增强版，包含更多测试用例 | Parquet |
| BigCodeBench | 大规模代码生成评估数据集 | Parquet |
| ClassEval | Python 类生成和测试基准 | Parquet |

---

## 实验结果

**TraceCoder 与基线方法在四个基准测试上的性能比较 (Pass@1, %)。"Ours" 表示本文提出的 TraceCoder 方法。每列最佳结果以粗体标记，括号内数值表示相对于第二名方法的绝对提升 (↑)。**

| Models | Methods | HumanEval | HumanEval+ | ClassEval | BigCodeBench-Complete | BigCodeBench-Instruct |
|:-------|:--------|:----------|:-----------|:----------|:----------------------|:----------------------|
| Gemini-2.5-Flash | Direct | 96.34 | 91.46 | 38.00 | 53.77 | 43.77 |
| | CoT | 93.90 | 91.46 | 41.00 | 53.86 | 43.68 |
| | Self-Planning | 94.51 | 90.85 | 36.00 | 55.61 | 43.15 |
| | Self-Debugging | 98.78 | 96.34 | 61.00 | 78.07 | 71.05 |
| | INTERVENOR | **99.39** | 95.12 | 61.00 | 75.88 | 69.82 |
| | **Ours** | **99.39 (↑0.61)** | **98.17 (↑1.83)** | **81.00 (↑20.00)** | **89.04 (↑10.97)** | **85.00 (↑13.95)** |
| DeepSeek-V3 | Direct | 94.51 | 90.24 | 41.00 | 38.25 | 46.67 |
| | CoT | 93.29 | 88.41 | 41.00 | 60.35 | 47.98 |
| | Self-Planning | 95.12 | 90.24 | 37.00 | 61.14 | 26.93 |
| | Self-Debugging | **98.78** | **96.34** | 61.00 | 82.37 | 74.56 |
| | INTERVENOR | 95.73 | 92.68 | 63.00 | 79.82 | 70.79 |
| | **Ours** | **98.78 (↑3.05)** | **96.34 (↑3.66)** | **78.00 (↑15.00)** | **88.33 (↑5.96)** | **83.77 (↑9.21)** |
| Qwen-Plus | Direct | 90.85 | 86.59 | 31.00 | 50.09 | 41.49 |
| | CoT | 93.29 | 87.19 | 33.00 | 48.07 | 43.50 |
| | Self-Planning | 90.85 | 84.75 | 37.00 | 37.36 | 41.75 |
| | Self-Debugging | **96.34** | **93.90** | 49.00 | 70.96 | 63.77 |
| | INTERVENOR | 95.12 | 91.46 | 48.00 | 68.60 | 61.75 |
| | **Ours** | **96.34 (↑1.22)** | **93.90 (↑2.44)** | **63.00 (↑14.00)** | **71.93 (↑0.97)** | **68.60 (↑4.83)** |

---

## 目录结构

```
TraceCoder/
├── config.py                 # 全局配置文件
├── trace_learn_coder.py      # 命令行主入口
├── problem_processor.py      # 核心问题处理逻辑
├── reporting.py              # 结果保存和统计
├── debug_api.py              # API 调试工具
├── requirements.txt          # 基础依赖
├── requirements-eval.txt     # 评估依赖
├── requirements-web.txt      # Web 界面依赖
├── start_web_demo.sh         # Web 服务启动脚本
│
├── src/                      # 核心模块
│   ├── dataset_loader.py     # 数据集加载
│   ├── generation.py         # 模型调用和代码生成
│   ├── traceRunner.py        # 代码执行和输出捕获
│   ├── postprocessing.py     # 代码后处理
│   └── simple_test_runner.py # 简单测试运行器
│
├── datasets/                 # 数据集评估模块
│   ├── human_eval/           # HumanEval 执行逻辑
│   ├── human_eval_plus/      # HumanEval+ 执行逻辑
│   ├── BigCodeBench/         # BigCodeBench 评估逻辑
│   └── ClassEval/            # ClassEval 评估逻辑
│
├── web_demo/                 # Web 图形界面
│   ├── app.py                # Flask 应用主文件
│   ├── trace_runner.py       # TraceCoder 流程运行器
│   ├── templates/            # HTML 模板
│   │   ├── base.html         # 基础模板
│   │   ├── home.html         # 首页
│   │   ├── generate.html     # 代码生成页
│   │   ├── history.html      # 历史记录页
│   │   ├── examples.html     # 示例库页
│   │   ├── settings.html     # 设置页
│   │   └── help.html         # 帮助页
│   └── static/               # 静态资源
│       ├── style.css         # 样式文件
│       └── app.js            # JavaScript 文件
│
└── results/                  # 实验结果输出目录
```

---

## 快速开始

### 1. 安装依赖

```bash
# 安装依赖
pip install -r requirements.txt
pip install -r requirements-web.txt
```

### 2. 启动 Web 界面

```bash
# 方式一：使用启动脚本
./start_web_demo.sh

# 方式二：直接运行
python web_demo/app.py
```

启动后在浏览器中访问 `http://localhost:5050`

### 3. 配置 API

在 Web 界面中点击导航栏的「设置」按钮，进行以下配置：

1. **输入 API 密钥**：填写您的 LLM API 密钥
2. **设置基础 URL**（可选）：如使用第三方 API 服务，填写对应的 API 端点
3. **获取模型列表**：点击「获取模型列表」按钮，选择要使用的模型
4. **验证配置**：点击「验证配置」按钮，确保 API 连接正常

### 4. 开始使用

1. 点击导航栏的「代码生成」进入主功能页面
2. 在「问题描述」区域输入编程问题
3. 在「测试用例」区域输入测试代码（assert 语句）
4. 点击「开始生成」按钮，系统将自动生成和调试代码
5. 实时查看生成过程、测试结果和调试步骤

> 💡 **提示**：您也可以点击「示例库」查看预定义的编程问题示例，支持一键导入。

---

## 命令行使用（高级）

如果您需要批量运行实验或使用数据集评估，可以使用命令行方式。

---

## Web 界面功能

### 页面概览

| 页面 | 功能描述 |
|------|----------|
| 首页 | 系统概览、功能特性介绍、快速开始指南 |
| 代码生成 | 核心功能页面，输入问题描述和测试用例，生成并调试代码 |
| 历史记录 | 查看所有历史任务，支持搜索、筛选和详情查看 |
| 示例库 | 预定义的编程问题示例，支持一键导入 |
| 设置 | API 配置、模型选择、运行参数设置 |
| 帮助 | 使用指南和常见问题解答 |

### 使用流程

1. **配置 API**：在设置页面输入 API 密钥和基础 URL
2. **选择模型**：获取可用模型列表并选择要使用的模型
3. **输入问题**：在代码生成页面输入问题描述和测试用例
4. **生成代码**：点击"开始生成"按钮，系统自动生成和调试代码
5. **查看结果**：实时查看生成过程、测试结果和调试步骤
6. **历史回顾**：在历史记录页面查看和管理所有任务

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 使用的大语言模型 | gemini-2.5-flash |
| `-d, --dataset` | 数据集名称 | humaneval |
| `--max-problems` | 最大处理问题数 | 全部 |
| `--max-debug-attempts` | 最大调试尝试次数 | 3 |
| `--timeout` | 代码执行超时时间（秒） | 10 |
| `--no-instrumentation` | 禁用代码插桩 | False |
| `--max-no-improvement-streak` | 无改善回滚阈值 | 2 |
| `--output-file` | 结果输出文件路径 | 自动生成 |

---

## 系统架构

TraceCoder 采用模块化的分层架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                  User Interface Layer                    │
│    (Flask Web Framework, HTML Templates, AJAX/SSE)      │
├─────────────────────────────────────────────────────────┤
│                 Business Logic Layer                     │
│  (trace_runner, problem_processor, generation, execution)│
├─────────────────────────────────────────────────────────┤
│                Data Processing Layer                     │
│       (dataset_loader, history_manager, storage)        │
├─────────────────────────────────────────────────────────┤
│                 External Service Layer                   │
│           (LLM API Client, OpenAI Compatible)           │
└─────────────────────────────────────────────────────────┘
```

### 核心模块说明

- **用户界面层**：提供 Web 图形界面，通过 AJAX 和 SSE 实现实时交互
- **业务逻辑层**：包含代码生成、执行测试、自动调试等核心逻辑
- **数据处理层**：负责数据集加载、历史记录管理和文件存储
- **外部服务层**：与大语言模型 API 进行交互

---

## 三步式调试机制

TraceCoder 的核心创新是三步式调试机制：

### 第一步：代码插桩 (Instrumentation)

对存在错误的代码进行分析，识别关键逻辑块，添加打印语句来追踪变量状态和执行流程：

```python
# 原始代码
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# 插桩后代码
def two_sum(nums, target):
    print(f"[TRACE] nums={nums}, target={target}")
    for i in range(len(nums)):
        print(f"[TRACE] i={i}, nums[i]={nums[i]}")
        for j in range(i + 1, len(nums)):
            print(f"[TRACE] j={j}, nums[j]={nums[j]}, sum={nums[i]+nums[j]}")
            if nums[i] + nums[j] == target:
                print(f"[TRACE] Found! returning [{i}, {j}]")
                return [i, j]
```

### 第二步：错误分析与修复计划 (Analysis & Planning)

执行插桩后的代码，捕获执行输出，结合测试失败信息，由大语言模型分析错误原因并制定详细的修复计划。

### 第三步：代码修复与验证 (Implementation)

根据修复计划生成修复后的代码，运行测试用例验证修复效果。如果修复后仍有错误，系统会进入调试循环。

---

## 安全机制

- **代码沙箱化**：使用独立子进程执行用户代码，限制危险操作
- **执行超时控制**：防止无限循环和长时间运行
- **资源使用限制**：防止恶意代码消耗过多资源
- **API 密钥本地存储**：密钥仅保存在浏览器本地，不上传服务器

---

## 依赖说明

### 基础依赖 (requirements.txt)
```
openai>=1.0.0
requests
```

### 评估依赖 (requirements-eval.txt)
```
datasets
pandas
pyarrow
tqdm
```

### Web 界面依赖 (requirements-web.txt)
```
flask
```

---

## 常见问题

### Q: API 连接失败怎么办？

A: 请检查以下几点：
1. API 密钥是否正确
2. 基础 URL 是否正确（如使用第三方 API）
3. 网络连接是否正常
4. 是否有 API 调用频率限制

### Q: 代码执行超时怎么办？

A: 可以在设置页面或命令行参数中增加超时时间：
- Web 界面：设置页面 → 运行参数 → 单次执行超时
- 命令行：`--timeout 30`

### Q: 如何添加新的数据集？

A: 需要以下步骤：
1. 在 `datasets/` 目录下创建评估模块
2. 在 `config.py` 中添加数据集配置
3. 在 `src/dataset_loader.py` 中添加加载逻辑

---

## 许可证

本项目采用 MIT 许可证。

---

## 致谢

感谢所有开源项目和数据集的贡献者，特别是：
- HumanEval 数据集
- BigCodeBench 数据集
- ClassEval 数据集
- OpenAI API
