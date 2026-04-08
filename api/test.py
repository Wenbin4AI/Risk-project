import requests

url = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "qwen3-vl:8b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "任务描述：\n请你作为一个建筑工程领域的专家，判断该图片所展示场景的风险类别并生成对应的风险描述。请按照指定的输出格式提供答案，不要输出其他内容。输出格式为：\n类别： [类别名称]\n简要说明： [简要描述图片内容与对应风险类别的匹配理由]\n输出示例：\n类别：临边、洞口防护不到位\n简要说明： 图片上方存在洞口，没有提供防护措施。\n下面是所有的分类：开挖深度2m及以上的基坑周边及坡道未按规范要求设置防护栏杆\n安全带或安全绳使用不符合要求\n临边、洞口防护不到位\n无风险\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://pic.rmb.bdstatic.com/bjh/news/5620d9f5adcdcc65fbad49e818998d38.png"
                    }
                }
            ]
        }
    ]
}

resp = requests.post(url, json=payload)
print(resp)
print(resp.json()["choices"][0]["message"]["content"])

