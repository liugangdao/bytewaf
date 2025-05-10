import torch
import torch.onnx


model_path = "model_save/model.pt"  # 替换为你的模型路径
model = torch.jit.load(model_path)
model.eval()

max_len = 100
# 创建一个示例输入张量，用于导出模型
# 假设你的模型需要输入大小为 (batch_size, 3, 224, 224) 的图像
dummy_input = torch.randint(0, 100, (1, max_len), dtype=torch.long).to(
    "cuda:0"
)  # 替换为你的输入大小

# 指定 ONNX 模型保存路径
onnx_model_path = "model_save/model.onnx"

# 将 PyTorch 模型导出为 ONNX 格式
torch.onnx.export(
    model,
    dummy_input,  # 示例输入
    onnx_model_path,  # 保存路径
    export_params=True,  # 导出模型参数
    opset_version=11,  # 使用 ONNX opset 版本 11
    input_names=["input"],  # 输入层名称
    output_names=["output"],  # 输出层名称
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)  # 支持动态 batch size

print(f"ONNX model saved to {onnx_model_path}")
