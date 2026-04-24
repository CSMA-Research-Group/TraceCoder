from flask import Flask, render_template, request, jsonify, Response
import os
import sys
import json
import requests
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_demo.trace_runner import TraceCoderRunner, running_status

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

trace_runner = TraceCoderRunner()

EXAMPLES = [
    {
        "id": "example1",
        "name": "两数之和",
        "description": "给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。",
        "test_case": "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\nassert two_sum([3, 2, 4], 6) == [1, 2]\nassert two_sum([3, 3], 6) == [0, 1]"
    },
    {
        "id": "example2",
        "name": "回文数判断",
        "description": "给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。",
        "test_case": "assert is_palindrome(121) == True\nassert is_palindrome(-121) == False\nassert is_palindrome(10) == False"
    },
    {
        "id": "example3",
        "name": "斐波那契数列",
        "description": "计算斐波那契数列的第n项。斐波那契数列定义为：F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2) (n > 1)",
        "test_case": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(2) == 1\nassert fibonacci(3) == 2\nassert fibonacci(4) == 3\nassert fibonacci(5) == 5"
    },
    {
        "id": "example4",
        "name": "快速排序",
        "description": "实现快速排序算法，对整数数组进行升序排列。要求处理各种边界情况，包括空数组、单元素数组、重复元素等。",
        "test_case": "assert quick_sort([]) == []\nassert quick_sort([1]) == [1]\nassert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]\nassert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]\nassert quick_sort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]"
    }
]

@app.route('/')
def index():
    """首页/仪表盘"""
    return render_template('home.html', active_page='home')

@app.route('/generate')
def generate_page():
    """代码生成页面"""
    return render_template('generate.html', active_page='generate')

@app.route('/history')
def history_page():
    """历史记录页面"""
    return render_template('history.html', active_page='history')

@app.route('/history/<task_id>')
def history_detail(task_id):
    """历史记录详情页面"""
    return render_template('history.html', active_page='history', task_id=task_id)

@app.route('/examples')
def examples_page():
    """示例库页面"""
    return render_template('examples.html', active_page='examples')

@app.route('/settings')
def settings_page():
    """设置页面"""
    return render_template('settings.html', active_page='settings')

@app.route('/help')
def help_page():
    """帮助页面"""
    return render_template('help.html', active_page='help')

@app.route('/api/run_tracecoder', methods=['POST'])
def run_tracecoder():
    """运行实际的TraceCoder流程"""
    try:
        data = request.json
        problem_description = data.get('problem_description')
        test_cases = data.get('test_cases')
        model_name = data.get('model_name', 'gemini-1.5-flash')
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        
        if not problem_description or not test_cases:
            return jsonify({'error': 'Missing problem_description or test_cases'}), 400
        
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
        
        configure_api_client(api_key, base_url)
        
        steps = list(trace_runner.run_problem_stream(problem_description, test_cases, model_name))
        
        return jsonify({'steps': steps})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_tracecoder_stream', methods=['POST'])
def run_tracecoder_stream():
    """流式运行实际的TraceCoder流程"""
    try:
        data = request.json
        problem_description = data.get('problem_description')
        test_cases = data.get('test_cases')
        model_name = data.get('model_name', 'gemini-1.5-flash')
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        
        if not problem_description or not test_cases:
            return jsonify({'error': 'Missing problem_description or test_cases'}), 400
        
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
        
        configure_api_client(api_key, base_url)
        
        def generate():
            try:
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
        
        configure_api_client(api_key, base_url)
        
        validation_result = test_api_connection(model_name, api_key, base_url)
        
        if validation_result['success']:
            return jsonify({'valid': True, 'message': 'API配置验证成功！'})
        else:
            return jsonify({'valid': False, 'error': validation_result['error']}), 400
            
    except Exception as e:
        return jsonify({'valid': False, 'error': f'验证过程中出现错误: {str(e)}'}), 500

@app.route('/api/examples')
def get_examples():
    """获取示例列表"""
    return jsonify(EXAMPLES)

@app.route('/api/examples/<example_id>')
def get_example(example_id):
    """获取单个示例详情"""
    for example in EXAMPLES:
        if example['id'] == example_id:
            return jsonify(example)
    return jsonify({'error': 'Example not found'}), 404

@app.route('/api/status')
def get_status():
    """获取系统状态"""
    return jsonify({
        'is_running': running_status.get('is_running', False),
        'version': '1.0.0'
    })

@app.route('/api/models', methods=['POST'])
def get_models():
    """获取可用的模型列表"""
    try:
        data = request.json
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400
        
        configure_api_client(api_key, base_url)
        
        # 尝试获取模型列表
        models = fetch_available_models(api_key, base_url)
        
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def fetch_available_models(api_key, base_url=None):
    """从API获取可用模型列表"""
    models = []
    
    try:
        # 根据 base_url 判断是哪个平台并调用相应的 API
        if base_url:
            if 'google' in base_url or 'gemini' in base_url:
                # Gemini API - 使用 Gemini 的模型列表端点
                models = fetch_gemini_models(api_key, base_url)
            elif 'anthropic' in base_url or 'claude' in base_url:
                # Claude API - Anthropic 目前没有公开的模型列表 API，使用固定列表
                models = fetch_anthropic_models(api_key, base_url)
            elif 'openai' in base_url or 'azure' in base_url:
                # OpenAI API - 使用 /models 端点
                models = fetch_openai_models(api_key, base_url)
            else:
                # 尝试使用 OpenAI 格式的 /models 端点
                models = fetch_openai_compatible_models(api_key, base_url)
        else:
            # 没有 base_url，默认使用 OpenAI 格式尝试
            models = fetch_openai_models(api_key, 'https://api.openai.com/v1')
    except Exception as e:
        print(f"Error fetching models: {e}")
        # 发生错误时返回空列表
        models = []
    
    return models

def fetch_openai_models(api_key, base_url):
    """获取 OpenAI 格式的模型列表"""
    models = []
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 构建 models 端点 URL
        models_url = base_url.rstrip('/') + '/models'
        
        response = requests.get(models_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', '')
                # 过滤掉非对话模型（如嵌入模型、图像模型等）
                if any(keyword in model_id.lower() for keyword in ['gpt', 'claude', 'gemini', 'chat', 'llama', 'qwen', 'mistral']):
                    models.append({
                        'id': model_id,
                        'name': model_id,
                        'description': model.get('description', '') or get_model_description(model_id)
                    })
        else:
            print(f"Failed to fetch models: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")
    
    return models

def fetch_openai_compatible_models(api_key, base_url):
    """获取 OpenAI 兼容格式的模型列表"""
    return fetch_openai_models(api_key, base_url)

def fetch_gemini_models(api_key, base_url):
    """获取 Gemini 模型列表"""
    models = []
    try:
        # Gemini 使用不同的 API 格式
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 尝试使用 Gemini 的 models 端点
        models_url = base_url.rstrip('/') + '/models'
        
        response = requests.get(models_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for model in data.get('models', []):
                model_name = model.get('name', '')
                if model_name:
                    # 提取模型 ID（去掉前缀）
                    model_id = model_name.split('/')[-1] if '/' in model_name else model_name
                    models.append({
                        'id': model_id,
                        'name': model_id,
                        'description': model.get('description', '') or get_model_description(model_id)
                    })
        else:
            # 如果 API 调用失败，使用 Gemini 的已知模型列表
            print(f"Gemini API 调用失败，使用默认模型列表")
            models = [
                {'id': 'gemini-1.5-flash', 'name': 'gemini-1.5-flash', 'description': '快速响应的多模态模型'},
                {'id': 'gemini-1.5-pro', 'name': 'gemini-1.5-pro', 'description': '高性能多模态模型'},
                {'id': 'gemini-1.0-pro', 'name': 'gemini-1.0-pro', 'description': '稳定可靠的文本模型'},
                {'id': 'gemini-pro', 'name': 'gemini-pro', 'description': 'Gemini Pro 模型'}
            ]
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")
        # 返回 Gemini 默认模型
        models = [
            {'id': 'gemini-1.5-flash', 'name': 'gemini-1.5-flash', 'description': '快速响应的多模态模型'},
            {'id': 'gemini-1.5-pro', 'name': 'gemini-1.5-pro', 'description': '高性能多模态模型'}
        ]
    
    return models

def fetch_anthropic_models(api_key, base_url):
    """获取 Anthropic/Claude 模型列表"""
    models = []
    try:
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # 尝试获取模型列表
        models_url = base_url.rstrip('/') + '/v1/models'
        
        response = requests.get(models_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', '')
                models.append({
                    'id': model_id,
                    'name': model_id,
                    'description': model.get('description', '') or get_model_description(model_id)
                })
        else:
            print(f"Anthropic API 调用失败，使用默认模型列表")
            # Anthropic 没有公开的模型列表 API，使用已知模型
            models = [
                {'id': 'claude-3-opus-20240229', 'name': 'claude-3-opus-20240229', 'description': '最强大的 Claude 3 模型'},
                {'id': 'claude-3-sonnet-20240229', 'name': 'claude-3-sonnet-20240229', 'description': '平衡性能和速度的模型'},
                {'id': 'claude-3-haiku-20240307', 'name': 'claude-3-haiku-20240307', 'description': '快速轻量级模型'},
                {'id': 'claude-3-5-sonnet-20241022', 'name': 'claude-3-5-sonnet-20241022', 'description': 'Claude 3.5 Sonnet 最新版'}
            ]
    except Exception as e:
        print(f"Error fetching Anthropic models: {e}")
        # 返回 Anthropic 默认模型
        models = [
            {'id': 'claude-3-sonnet-20240229', 'name': 'claude-3-sonnet-20240229', 'description': '平衡性能和速度的模型'},
            {'id': 'claude-3-opus-20240229', 'name': 'claude-3-opus-20240229', 'description': '最强大的 Claude 3 模型'}
        ]
    
    return models

def get_model_description(model_id):
    """根据模型 ID 获取描述"""
    descriptions = {
        # OpenAI
        'gpt-4': 'OpenAI GPT-4 旗舰模型',
        'gpt-4-turbo': 'GPT-4 Turbo 增强版',
        'gpt-4-turbo-preview': 'GPT-4 Turbo 预览版',
        'gpt-4o': 'GPT-4o 多模态模型',
        'gpt-4o-mini': 'GPT-4o Mini 轻量版',
        'gpt-3.5-turbo': 'GPT-3.5 Turbo 高性价比',
        # Gemini
        'gemini-1.5-flash': 'Gemini 1.5 Flash 快速响应',
        'gemini-1.5-pro': 'Gemini 1.5 Pro 高性能',
        'gemini-1.0-pro': 'Gemini 1.0 Pro 稳定版',
        'gemini-pro': 'Gemini Pro 标准版',
        # Claude
        'claude-3-opus-20240229': 'Claude 3 Opus 最强模型',
        'claude-3-sonnet-20240229': 'Claude 3 Sonnet 平衡版',
        'claude-3-haiku-20240307': 'Claude 3 Haiku 快速版',
        'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet 最新版',
        # 通用
        'default': '默认模型'
    }
    
    # 尝试精确匹配
    if model_id in descriptions:
        return descriptions[model_id]
    
    # 尝试部分匹配
    for key, desc in descriptions.items():
        if key in model_id.lower():
            return desc
    
    return 'AI 语言模型'

def configure_api_client(api_key, base_url=None):
    """配置API客户端"""
    try:
        from src.generation import get_client
        client = get_client()
        client.api_key = api_key
        if base_url:
            client.base_url = base_url
    except ImportError:
        pass

def test_api_connection(model_name, api_key, base_url=None):
    """测试API连接"""
    try:
        from src.generation import get_client
        client = get_client()
        
        client.api_key = api_key
        if base_url:
            client.base_url = base_url
            
        print(f"Testing API connection with:")
        print(f"  API Key: {api_key[:8]}..." if api_key else "None")
        print(f"  Base URL: {base_url}")
        print(f"  Model: {model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello, this is a test message to verify the API connection."}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        if response and response.choices and len(response.choices) > 0:
            return {'success': True}
        else:
            return {'success': False, 'error': 'API响应无效或为空'}
    except Exception as e:
        error_msg = str(e)
        print(f"API Connection Error: {error_msg}")
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
