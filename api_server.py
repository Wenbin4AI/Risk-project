import json
import redis
from flask import Flask, request, jsonify
import logging
import datetime
import os
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# 连接 Redis
r = redis.Redis(host='localhost', port=6379, db=0)

QUEUE_NAME = "image_tasks"

def save_data(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    if not data.get("image_url") or not data.get("callback_url"):
        return jsonify({"error": "image_url and callback_url required"}), 400

    # 生成唯一文件名
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y%m%d_%H%M%S')
    random_filename = f"{time_str}_{uuid.uuid4().hex}"

    folder_path = "./ask"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filepath = os.path.join(folder_path, f"{random_filename}.json")
    save_data(filepath, data)

    # 推入 Redis 队列
    r.lpush(QUEUE_NAME, json.dumps(data))

    return jsonify({"status": "accepted"}), 202


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8070)