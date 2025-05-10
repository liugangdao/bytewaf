FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv curl && \
    ln -sf python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 包
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir onnxruntime-gpu==1.20.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
# 拷贝代码
WORKDIR /app
COPY app.py /app/app.py


# 运行服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5555", "--workers", "8"]
