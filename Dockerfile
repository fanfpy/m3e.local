# auth email fanfpy@qq.com
# 使用官方的Python 3.10镜像作为基础镜像
FROM python:3.10-slim-bullseye

# 设置 pip 源为清华大学的镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY /app /app/

# 安装git-lfs
RUN apt-get update
RUN apt-get install -y git-lfs

# 在镜像中执行git clone 下载模型 国内源
RUN git lfs install
RUN git clone https://www.modelscope.cn/Jerry0/M3E-large.git /app/M3E-large


# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口（如果你的应用需要监听端口）
EXPOSE 6006

# 设置启动命令
CMD ["python", "m3e-openapi.py"]
