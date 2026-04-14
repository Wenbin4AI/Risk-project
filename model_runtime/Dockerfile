FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    git curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY download_model.py /app/download_model.py
COPY server.py /app/server.py

ENV MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
ENV MODEL_PATH=/models/Qwen/Qwen2.5-VL-7B-Instruct

RUN mkdir -p /models

EXPOSE 9000

CMD ["bash", "-c", "python3 /app/download_model.py && uvicorn server:app --host 0.0.0.0 --port 9000"]