import pytest
from fastapi.testclient import TestClient
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.app import app  # 假设你的 FastAPI 应用程序名为 app.py

# 创建测试客户端
client = TestClient(app)

# 模拟输入数据
test_data = {
    "texts": [
        "id=1 OR 1=1",  # SQLi
        "<script>alert('x')</script>",  # XSS
        "page=home",  # 正常请求
        "script",
    ],
    "max_len": 100,
}


# 测试案例
def test_predict():
    response = client.post("/detect", json=test_data)

    # 断言状态码
    assert response.status_code == 200

    # 断言返回的结果字段
    response_json = response.json()
    assert "max_score" in response_json
    assert "label" in response_json
    assert "text" in response_json
    print(response_json)
    # 断言标签和分数
    assert response_json["label"] in ["攻击 🚨", "正常 ✅"]
    assert isinstance(response_json["max_score"], float)
    assert isinstance(response_json["text"], str)


# 可选: 测试输入验证错误（例如缺少 text 字段）
def test_missing_text_field():
    invalid_data = {"max_len": 10}
    response = client.post("/detect", json=invalid_data)

    assert response.status_code == 422  # HTTP 422: Unprocessable Entity, 用于验证错误
