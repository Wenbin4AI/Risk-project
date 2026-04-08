import requests
import json

url = "http://localhost:8070/process-image"

data = {
    "callback_url": "http://127.0.0.1:36801/api/notify/processImage/reportNotify/1989152798487511041",
    "image_url": "https://blog.ansi.org/wp-content/uploads/2024/05/GettyImages-2147639892-unprotected-edge.jpg",
    "objectKey": "VID_20251115_093522_02/change/1989152798487511041",
    "device_type": "X4",
    "lang": "ch",
    "image_type": "02"
}

response = requests.post(url, json=data)

print("状态码:", response.status_code)
print("返回内容:", response.json())


# import requests

# url = "http://localhost:8070/process-image"

# for i in range(10):
#     data = {
#         "image_url": f"test_image_{i}.jpg",
#         "callback_url": "http://localhost:9000/callback"
#     }

#     response = requests.post(url, json=data)
#     print(f"任务 {i} 状态:", response.status_code)