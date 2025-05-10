import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
import tiktoken
from typing import List
import logging

# 设置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        # logging.FileHandler("app.log")  # 可选：写入文件
    ],
)

logger = logging.getLogger("onnx_service")

# 初始化 FastAPI 应用
app = FastAPI()

# 加载 ONNX 模型
session = ort.InferenceSession(
    "model_save/model.onnx", providers=["CUDAExecutionProvider"]
)

# 加载 tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


# 定义输入数据格式
class InputData(BaseModel):
    texts: List[str]
    max_len: int


# 创建预测接口
@app.post("/detect")
def predict(data: InputData):
    texts = data.texts
    max_len = data.max_len
    # 对输入文本进行编码
    all_tokens = []
    # for t in text:
    #     tokens = tokenizer._encode_bytes(t)
    #     all_tokens.append(tokens[:max_len] + [0] * (max_len - len(tokens)))
    start = time.time()
    all_tokens = tokenizer.encode_batch(texts)
    for i in range(len(all_tokens)):
        all_tokens[i] = all_tokens[i][:max_len] + [0] * (max_len - len(all_tokens[i]))

    # 转换为 numpy 数组
    input_tensor = np.array(all_tokens, dtype=np.int64)
    # print(input_tensor.shape)
    # 执行批量推理

    outputs = session.run(None, {"input": input_tensor})
    logger.info("推理耗时：", time.time() - start)
    # 获取分数和标签
    # print(outputs)
    scores = outputs[0].squeeze(axis=1)  # shape: (batch_size,) 一维
    labels = np.where(scores > 0.5, "攻击 🚨", "正常 ✅")

    results = [
        {"text": text, "score": float(score), "label": label}
        for text, score, label in zip(data.texts, scores, labels)
    ]

    # 找到分数最大的结果
    max_idx = np.argmax(scores)
    max_result = results[max_idx]

    return {
        "max_score": max_result["score"],
        "label": max_result["label"],
        "text": max_result["text"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5555, workers=8)
