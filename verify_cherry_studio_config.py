import requests

# 模拟CherryStudio的配置
cherry_studio_endpoint = "http://127.0.0.1:7860/v1/chat/completions"
api_key = "123"

print("验证CherryStudio配置...")
print(f"完整端点: {cherry_studio_endpoint}")
print(f"API密钥: {api_key}")

# 尝试直接调用CherryStudio的完整端点
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "gemini-2.5-flash",  # 使用您实际的模型名称
    "messages": [
        {"role": "user", "content": "Hello, this is a test message to verify the API connection."}
    ],
    "max_tokens": 10,
    "temperature": 0.1
}

try:
    response = requests.post(cherry_studio_endpoint, headers=headers, json=data, timeout=30)
    print(f"\n直接调用结果:")
    print(f"  状态码: {response.status_code}")
    if response.status_code == 200:
        print("  调用成功!")
        print(f"  响应: {response.text[:200]}...")
    else:
        print(f"  调用失败: {response.text}")
except Exception as e:
    print(f"\n直接调用出错: {e}")

print("\n" + "="*50)

# 现在测试我们的配置方式（基础URL + 模型）
base_url = "http://127.0.0.1:7860/v1"
print(f"我们的配置方式:")
print(f"  基础URL: {base_url}")
print(f"  API密钥: {api_key}")
print(f"  模型: gemini-2.5-flash")

# 我们的客户端会自动拼接/chat/completions路径
chat_url = f"{base_url}/chat/completions"
print(f"  实际调用URL: {chat_url}")

try:
    response = requests.post(chat_url, headers=headers, json=data, timeout=30)
    print(f"\n我们的配置方式调用结果:")
    print(f"  状态码: {response.status_code}")
    if response.status_code == 200:
        print("  调用成功!")
        print(f"  响应: {response.text[:200]}...")
    else:
        print(f"  调用失败: {response.text}")
except Exception as e:
    print(f"\n我们的配置方式调用出错: {e}")