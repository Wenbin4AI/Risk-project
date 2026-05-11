# 异步进程（ Redis 队列 + Worker  ）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n llm python=3.10
conda activate llm

# 安装 Redis
sudo apt install redis-server
启动：
sudo systemctl start redis
测试：
redis-cli ping
返回：
PONG
安装python依赖
pip install redis flask requests
pip install torch opencv-python-headless Pillow
pip install deep-translator openai

### LLM api
cd /home/synloop/risk/api
conda activate api
uvicorn server:app --host 0.0.0.0 --port 8000

### risk program server
cd /home/synloop/risk
conda activate llm
python api_server.py

python worker.py