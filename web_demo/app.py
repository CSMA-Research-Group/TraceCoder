from flask import Flask, render_template, request, jsonify, Response
import os
import sys
import json
import requests
import time
import threading

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 trace_simulator 导入改为从 trace_runner 导入
from web_demo.trace_runner import TraceCoderRunner, running_status

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# 初始化 TraceCoderRunner 而不是模拟器
trace_runner = TraceCoderRunner()

@app.route('/')
def index():
    """主页"""
    # 返回一个空的示例列表，因为我们现在支持自定义问题
    return render_template('index.html', examples=[])

@app.route('/api/run_tracecoder', methods=['POST'])
def run_tracecoder():
    """运行实际的TraceCoder流程"""
    try:
        data = request.json
        # 获取自定义问题描述和测试用例，而不是示例ID
        problem_description = data.get('problem_description')
        test_cases = data.get('test_cases')
        model_name = data.get('model_name', 'gemini-1.5-flash')
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        
        if not problem_description or not test_cases:
            return jsonify({'error': 'Missing problem_description or test_cases'}), 400
        
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
        
        # 配置API密钥和基础URL
        configure_api_client(api_key, base_url)
        
        # 运行实际的TraceCoder流程
        steps = list(trace_runner.run_problem_stream(problem_description, test_cases, model_name))
        
        return jsonify({'steps': steps})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_tracecoder_stream', methods=['POST'])
def run_tracecoder_stream():
    """流式运行实际的TraceCoder流程"""
    try:
        data = request.json
        # 获取自定义问题描述和测试用例，而不是示例ID
        problem_description = data.get('problem_description')
        test_cases = data.get('test_cases')
        model_name = data.get('model_name', 'gemini-1.5-flash')
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        
        if not problem_description or not test_cases:
            return jsonify({'error': 'Missing problem_description or test_cases'}), 400
        
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
        
        # 配置API密钥和基础URL
        configure_api_client(api_key, base_url)
        
        def generate():
            try:
                # 流式运行TraceCoder流程
                for step in trace_runner.run_problem_stream(problem_description, test_cases, model_name):
                    yield f"data: {json.dumps(step)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_tracecoder', methods=['POST'])
def stop_tracecoder():
    """停止当前运行的TraceCoder流程"""
    try:
        # 设置停止标志
        running_status["should_stop"] = True
        
        return jsonify({'message': '已发送停止请求'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate_config', methods=['POST'])
def validate_config():
    """验证API配置"""
    try:
        data = request.json
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        model_name = data.get('model_name', 'gemini-1.5-flash')
        
        if not api_key:
            return jsonify({'valid': False, 'error': 'Missing API key'}), 400
        
        # 配置API客户端
        configure_api_client(api_key, base_url)
        
        # 尝试进行一个简单的API调用来验证配置
        validation_result = test_api_connection(model_name, api_key, base_url)
        
        if validation_result['success']:
            return jsonify({'valid': True, 'message': 'API配置验证成功！'})
        else:
            return jsonify({'valid': False, 'error': validation_result['error']}), 400
            
    except Exception as e:
        return jsonify({'valid': False, 'error': f'验证过程中出现错误: {str(e)}'}), 500

def configure_api_client(api_key, base_url=None):
    """配置API客户端"""
    try:
        # 导入generation模块并配置客户端
        from src.generation import get_client
        client = get_client()
        client.api_key = api_key
        if base_url:
            client.base_url = base_url
    except ImportError:
        pass  # 如果无法导入，则跳过配置

def test_api_connection(model_name, api_key, base_url=None):
    """测试API连接"""
    try:
        # 检查API密钥是否已设置
        from src.generation import get_client
        client = get_client()
        
        # 配置客户端
        client.api_key = api_key
        if base_url:
            client.base_url = base_url
            
        # 输出调试信息
        print(f"Testing API connection with:")
        print(f"  API Key: {api_key[:8]}..." if api_key else "None")
        print(f"  Base URL: {base_url}")
        print(f"  Model: {model_name}")
        
        # 尝试进行一个简单的调用来验证连接
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello, this is a test message to verify the API connection."}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        # 检查响应是否有效
        if response and response.choices and len(response.choices) > 0:
            return {'success': True}
        else:
            return {'success': False, 'error': 'API响应无效或为空'}
    except Exception as e:
        # 提供更详细的错误信息
        error_msg = str(e)
        print(f"API Connection Error: {error_msg}")  # 调试输出
        
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {'success': False, 'error': 'API密钥验证失败，请检查您的API密钥是否正确'}
        elif "model" in error_msg.lower():
            return {'success': False, 'error': f'模型"{model_name}"不可用，请检查模型名称是否正确'}
        elif "rate limit" in error_msg.lower():
            return {'success': False, 'error': 'API调用频率超限，请稍后再试'}
        elif "timeout" in error_msg.lower():
            return {'success': False, 'error': 'API调用超时，请检查网络连接'}
        elif "404" in error_msg:
            return {'success': False, 'error': 'API端点未找到，请检查基础URL是否正确'}
        elif "connection" in error_msg.lower():
            return {'success': False, 'error': '网络连接失败，请检查网络设置和防火墙'}
        else:
            return {'success': False, 'error': f'API连接失败: {error_msg}'}

@app.route('/api/examples')
def get_examples():
    """获取示例列表 - 返回空列表因为我们支持自定义问题"""
    return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)