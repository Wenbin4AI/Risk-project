import os
import time
import uuid
from io import BytesIO
from typing import List, Optional

import requests
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

app = FastAPI()

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")

model = None
processor = None


class GenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    max_new_tokens: int = 256


def load_image(image_url: str) -> Image.Image:
    if image_url.startswith("http://") or image_url.startswith("https://"):
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        if not os.path.exists(image_url):
            raise FileNotFoundError(f"Image not found: {image_url}")
        return Image.open(image_url).convert("RGB")


@app.on_event("startup")
def startup_event():
    global model, processor

    # 找到 ModelScope 下载后的目录
    candidate_dir = MODEL_ID
    local_candidate = os.path.join(MODEL_DIR, "hub", "models", MODEL_ID)
    if os.path.exists(local_candidate):
        candidate_dir = local_candidate

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        candidate_dir,
        torch_dtype="auto" if TORCH_DTYPE == "auto" else getattr(torch, TORCH_DTYPE),
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(candidate_dir)


@app.get("/health")
def health():
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="model not loaded")
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        if req.image_url:
            messages[0]["content"].append({"type": "image", "image": req.image_url})

        messages[0]["content"].append({"type": "text", "text": req.prompt})

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return {
            "id": f"gen-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": MODEL_ID,
            "text": output_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))