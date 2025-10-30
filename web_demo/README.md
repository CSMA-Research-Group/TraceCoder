# TraceCoder Web Demo

这是一个用于可视化展示TraceCoder工作流程的Web演示应用。

## 目录结构

```
web_demo/
├── app.py                 # Flask应用主文件
├── trace_simulator.py     # TraceCoder流程模拟器
├── examples_config.json   # 示例问题配置
├── templates/             # HTML模板
│   └── index.html        # 主页面模板
├── static/               # 静态资源文件
│   └── style.css         # 自定义样式
└── README.md             # 本文件
```

## 功能特点

1. **可视化流程**：清晰展示TraceCoder的完整工作流程
2. **交互式示例选择**：提供预定义的编程问题供用户选择
3. **实时步骤展示**：动态显示每个步骤的执行状态和详细信息
4. **代码展示**：以高亮格式显示生成和修复的代码
5. **流程图展示**：可视化TraceCoder的14步流程图

## 启动应用

```bash
# 方法1：直接运行
python web_demo/app.py

# 方法2：使用启动脚本
./start_web_demo.sh
```

## 访问应用

启动应用后，在浏览器中访问：http://localhost:5050

## 技术栈

- Flask：Web框架
- Bootstrap：前端UI框架
- JavaScript：前端交互逻辑
- HTML/CSS：页面结构和样式

## 自定义配置

可以通过修改 `examples_config.json` 文件来添加更多示例问题：

```json
{
    "examples": [
        {
            "id": "example1",
            "name": "示例名称",
            "description": "问题描述",
            "test_case": "测试用例"
        }
    ]
}
```