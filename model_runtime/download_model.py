import os
from modelscope import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

if __name__ == "__main__":
    local_path = snapshot_download(
        model_id=MODEL_ID,
        cache_dir=MODEL_DIR
    )
    print(f"Model downloaded to: {local_path}")