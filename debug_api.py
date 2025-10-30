import sys
import os
import requests
import socket
from openai import OpenAI

def check_port_open(host, port):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False

def test_service_accessibility(host, port):
    """测试服务可访问性"""
    print(f"检查 {host}:{port} 是否可访问...")
    
    # 检查端口是否开放
    if check_port_open(host, port):
        print(f"  端口 {port} 是开放的")
    else:
        print(f"  端口 {port} 被拒绝连接")
        return False
    
    # 尝试访问根路径
    try:
        response = requests.get(f"http://{host}:{port}", timeout=10)
        print(f"  根路径响应状态码: {response.status_code}")
        return True
    except Exception as e:
        print(f"  访问根路径失败: {e}")
        return False

def test_api_connection_requests(base_url, api_key, model_name):
    """使用requests库直接测试API连接"""
    try:
        print(f"\nTesting API connection with requests:")
        print(f"  Base URL: {base_url}")
        print(f"  API Key: {api_key[:3]}***")
        print(f"  Model: {model_name}")
        
        # 构建完整的聊天完成URL
        if base_url.endswith('/'):
            chat_url = base_url + "chat/completions"
        else:
            chat_url = base_url + "/chat/completions"
        
        print(f"  Full URL: {chat_url}")
        
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 设置请求数据
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message to verify the API connection."}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        # 发送POST请求
        response = requests.post(chat_url, headers=headers, json=data, timeout=30)
        
        print(f"  Status Code: {response.status_code}")
        if response.status_code == 200:
            print("API连接成功！")
            return True
        else:
            print(f"  Response: {response.text[:200]}...")
            print(f"API调用失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"Requests连接失败: {e}")
        return False

if __name__ == "__main__":
    # 配置信息
    host = "127.0.0.1"
    port = 7860
    base_url = f"http://{host}:{port}/v1"
    api_key = "123"
    model_name = "gemini-2.5-flash"  # 请根据实际情况修改
    
    print("开始调试API连接问题...")
    print("=" * 50)
    
    # 1. 检查服务可访问性
    if not test_service_accessibility(host, port):
        print("\n服务不可访问，请检查:")
        print("1. 确保您的API服务正在运行")
        print("2. 检查是否有防火墙阻止连接")
        print("3. 确认服务监听在正确的地址和端口")
        sys.exit(1)
    
    # 2. 尝试API连接
    print("\n" + "=" * 50)
    success = test_api_connection_requests(base_url, api_key, model_name)
    
    if success:
        print("\n测试成功！API连接正常。")
    else:
        print("\n测试失败！请检查配置。")