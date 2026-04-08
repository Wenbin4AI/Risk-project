from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY"  # vLLM 不校验
)

response = client.chat.completions.create(
    model="/home/synloop/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B",  # 你启动时的 --served-model-name
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)