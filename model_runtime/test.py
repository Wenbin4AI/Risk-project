import requests
import json

BASE_URL = "http://127.0.0.1:9000"

def test_health():
    url = f"{BASE_URL}/health"
    resp = requests.get(url, timeout=30)
    print("health status code:", resp.status_code)
    print("health response:", resp.text)
    resp.raise_for_status()

def test_text_only():
    url = f"{BASE_URL}/generate"
    payload = {
        "prompt": "Please briefly introduce yourself in one sentence.",
        "max_new_tokens": 64
    }
    resp = requests.post(url, json=payload, timeout=120)
    print("text-only status code:", resp.status_code)
    print("text-only response:")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    resp.raise_for_status()

def test_with_image():
    url = f"{BASE_URL}/generate"
    payload = {
        "prompt": "Describe this image briefly.",
        "image_url": "https://picsum.photos/400/300",
        "max_new_tokens": 64
    }
    resp = requests.post(url, json=payload, timeout=180)
    print("image status code:", resp.status_code)
    print("image response:")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    resp.raise_for_status()

if __name__ == "__main__":
    test_health()
    test_text_only()
    test_with_image()