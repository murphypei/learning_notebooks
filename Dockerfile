# 使用指定的基础镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 安装必要的软件包
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    curl \
    wget

RUN pip install \
    datasets \
    trl \
    peft \
    bitsandbytes \
    wandb \
    accelerate \
    transformers \
    ipywidgets

# RUN git clone https://github.com/vllm-project/vllm.git && \
#     cd vllm && \
#     pip install -e .

# 设置工作目录
WORKDIR /ws
