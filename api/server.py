import ollama
import base64
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import os
import time

app = FastAPI()


# ========= 工具函数 =========

def image_to_base64(path_or_url: str) -> str:
    # 情况 1：HTTP / HTTPS URL
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")

    # 情况 2：本地路径
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Image not found: {path_or_url}")
        img = Image.open(path_or_url).convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_qwen(prompt: str, image_b64: str | None = None) -> str:
    message = {
        "role": "user",
        "content": prompt,
    }

    if image_b64 is not None:
        message["images"] = [image_b64]

    response = ollama.chat(
        model="qwen3-vl:8b",
        messages=[message],
    )

    return response["message"]["content"]


# ========= OpenAI 风格请求体 =========

class ImageURL(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None


class Message(BaseModel):
    role: str
    content: list[ContentItem]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]


# ========= API =========

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        user_msg = req.messages[-1]

        prompt_parts = []
        image_url = None

        for item in user_msg.content:
            if item.type == "text" and item.text:
                prompt_parts.append(item.text)
            elif item.type == "image_url" and item.image_url:
                image_url = item.image_url.url

        prompt = "".join(prompt_parts)

        image_b64 = None
        if image_url:
            image_b64 = image_to_base64(image_url)

        answer = call_qwen(prompt, image_b64)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

