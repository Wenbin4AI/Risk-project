from cubemap_generator_torch import create_cubemap_generator_torch
from PIL import Image
from flask import Flask, request, jsonify
import requests
from deep_translator import GoogleTranslator
from openai import OpenAI
import json
import os
import time 
import logging
import base64
import datetime
import torch
import re


corrected_generator = create_cubemap_generator_torch()

def extract_direction(path: str):
    match = re.search(r'(back|front|left|right)', path)
    return match.group(1) if match else None

# 下载图片并保存到本地文件夹
def download_image(image_url):
    try:
        # 确保图片保存文件夹存在
        picture_folder = "save_original_picture"
        if not os.path.exists(picture_folder):
            os.makedirs(picture_folder)

        # 下载图片
        response = requests.get(image_url)
        if response.status_code == 200:
            # 获取图片的文件名
            image_name = os.path.basename(image_url)
            image_path = os.path.join(picture_folder, image_name)

            # 保存图片
            with open(image_path, 'wb') as f:
                f.write(response.content)

            return image_path  # 返回图片的保存路径
        else:
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# 畸变校正
def distortion_correction_six(image_url, output_url, resolution=2048):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "save_corrected_picture")
    output_subdir = os.path.join(output_dir, output_url)
    os.makedirs(output_subdir, exist_ok=True)

    
    # 运行立方体贴图生成算法
    result = corrected_generator.generate_cubemap_batch(
        panorama_image=image_url,
        output_dir=str(output_subdir),
        resolution=resolution
    )
    
    return result

# save result
def save_json_to_results(data, filename):
    # 确保results文件夹存在
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 构建完整的文件路径
    file_path = os.path.join(results_dir, f"{filename}.json")

    try:
        # 将数据写入文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到 {file_path}")
        return True
    except Exception as e:
        print(f"保存文件时发生错误: {e}")
        return False

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 压缩图片
def compress_image(image_path, output_path, quality=50):
    # 打开原始图片
    with Image.open(image_path) as img:
        # 如果不是 RGB 模式，转换为 RGB 以兼容 JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 保存压缩后的图片
        img.save(output_path, format="JPEG", quality=quality)
        print(f"图片已压缩并保存到 {output_path}")

# 翻译成对应语言
def translate_entry(entry, target_lang):
    translated_entry = entry.copy()
    fields_to_translate = ["dangerType", "dangerContent"]
    for field in fields_to_translate:
        try:
            translated_entry[field] = GoogleTranslator(source='zh-CN', target=target_lang).translate(entry[field])
        except Exception as e:
            print(f"翻译字段 {field} 时出错: {str(e)}")
    return translated_entry

# 获取当前时间的名称
def get_name():
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%Y%m%d_%H%M%S')
    file_name = f'{time_str}'  # 你可以将 .txt 替换为其他扩展名，如 .csv、.log 等
    return file_name

# 将模型结果转化为json格式
def convert_to_json(text, image_path, ans="_1.jpg", is_pano=False):

    # 将图片大小压缩
    base64_image = image_to_base64(image_path)
    # print(f"原图片Base64字符串长度: {len(base64_image)}")

    original_image_path = image_path
    compressed_image_path = image_path + ".jpg"
    compression_quality = 40
    compress_image(original_image_path, compressed_image_path, quality=compression_quality)

    base64_image = image_to_base64(compressed_image_path)
    # print(f"压缩后图片Base64字符串长度: {len(base64_image)}")

    text["image"] = base64_image

    return text

# 调用本地 LLM
def get_result_local(image="", prompt=""):
    if prompt == "":
        prompt = (
            "你是一个建筑施工安全巡检领域的专业工程师。\n"
            "请基于输入的图片判断是否存在建筑施工安全风险。\n\n"
            "必须严格遵守以下输出规则：\n"
            "1. 仅输出一个 JSON 对象，不得输出任何额外文本、说明、Markdown、注释或换行说明。\n"
            "2. JSON 只能包含以下两个字段：\"风险名称\" 和 \"风险描述\"。\n"
            "3. 如果未发现任何施工安全风险：\n"
            "   - \"风险名称\" 必须为 \"无风险\"\n"
            "   - \"风险描述\" 必须为 \"未发现明显施工安全风险\"\n"
            "4. 如果存在多个风险，仅输出最严重的一个。\n\n"
            "JSON 输出示例：\n"
            "{\"风险名称\":\"未佩戴安全帽\",\"风险描述\":\"施工现场作业人员未佩戴安全帽，存在头部受伤风险。\"}"
        )
    url = "http://127.0.0.1:8000/v1/chat/completions"
    if image=="":
        payload = {
            "model": "qwen3-vl:8b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }
    else:
        payload = {
            "model": "qwen3-vl:8b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image
                            }
                        }
                    ]
                }
            ]
        }

    resp = requests.post(url, json=payload)
    return resp.json()["choices"][0]["message"]["content"]

# 调用远程 LLM
def get_result_api(image_path, prompt=""):
    example_a = {"danger_class" : "", "danger_describe" : ""}
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key="sk-e26dbd1a0cb64659bd80da8a50bea222",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    if prompt == "":
        prompt = f"请你结合上传的图片内容，从专业角度判断是否存在以下4类建筑风险隐患，并对该风险进行80字以内的描述说明具体风险位置和状态：\n可选风险类型如下（如存在，请从中选择一类填写）：\n临时用电\n临边、洞口防护不到位\n脚手架防护不到位\n无风险\n请严格按照以下格式返回你的分析结果（必须是 JSON 对象，不要输出其他任何内容）：\n" + json.dumps(example_a, ensure_ascii=False, indent=4) + "\n注意：不要输出除 JSON 对象之外的任何解释、注释或额外文字。"

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一位建筑安全巡查专家。"},
            {"role": "user", "content": prompt},
        ]
    )
    # print(completion.model_dump_json())

    return completion.model_dump_json()

def extract_assistant_json(text: str):
    text = text.strip()

    # 情况 1：模型直接返回纯 JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}\n原始内容:\n{text}")

    # 情况 2：包含 assistant 前缀
    match = re.search(
        r"assistant\s*(\{.*\})",
        text,
        flags=re.DOTALL
    )

    if not match:
        raise ValueError(f"未找到可解析的 JSON 内容:\n{text}")

    json_str = match.group(1).strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}\n原始 JSON 内容:\n{json_str}")

def _no_risk_result():
    return {
        "风险名称": "无风险",
        "风险描述": "未发现明显施工安全风险"
    }


def run(image_path: str):
    # ========== Step 1 ==========
    try:
        step1_output = get_result_local(image=image_path)
        print(step1_output)
        risk = extract_assistant_json(step1_output)
    except Exception as e:
        logging.info(f"[WARN] Step1 失败，自动降级为无风险: {e}")
        step1_output = ""
        current_time = datetime.datetime.now()
        file_name = current_time.strftime('%Y%m%d_%H%M%S')
        save_json_to_results({"fjaf":"fjak"}, file_name)

        risk = _no_risk_result()

    # ========== Step 2 ==========
    try:
        # 如果第一步已经是“无”，直接跳过 LLM
        if risk["风险名称"] == "无风险":
            risk2 = _no_risk_result()
            step2_output = ""
        else:
            prompt = (
                "你是一个建筑施工安全巡检领域的专业风险分类模型。\n\n"
                "你的任务是：根据给定的风险描述，从【固定枚举的风险类型】中严格选择且仅选择一个最符合的类别。\n\n"
                "【可选风险类型枚举（只能从中选择一个，不得修改、不许扩展）】：\n"
                "1. 未带安全带/安全绳/生命线\n"
                "2. 未带安全帽\n"
                "3. 临边、洞口防护不足\n"
                "4. 现场脏乱\n"
                "5. 现场材料堆放不合理\n"
                "6. 现场积水\n"
                "7. 电线拖地、泡水\n"
                "8. 随意吸烟、明火、烟雾\n"
                "9. 动火作业邻近易燃、易爆物品\n"
                "10. 材料堆放过高\n"
                "11. 脚手架未设置斜撑\n"
                "12. 基坑周边未设置防护杆\n"
                "13. 无风险\n\n"
                "【分类判定规则（必须严格遵守）】：\n"
                "（1）如描述中明确提及高处作业，且存在未佩戴或未正确使用安全带、安全绳或生命线的情况，\n"
                "     无论是否同时存在其他风险，优先选择：\"未带安全带/安全绳/生命线\"。\n"
                "（2）如描述涉及人员未佩戴安全帽，且未涉及第（1）条所述情况，选择：\"未带安全帽\"。\n"
                "（3）如描述涉及临边、洞口，且防护设施缺失、破损或设置不规范，选择：\"临边、洞口防护不足\"。\n"
                "（4）如描述涉及基坑作业，且周边未设置防护栏杆或防护杆，选择：\"基坑周边未设置防护杆\"。\n"
                "（5）如描述涉及脚手架结构问题，且明确提及未设置斜撑，选择：\"脚手架未设置斜撑\"。\n"
                "（6）如描述涉及施工现场材料堆放高度明显超标，存在倾倒、坠落风险，选择：\"材料堆放过高\"。\n"
                "（7）如描述涉及材料堆放杂乱、占道、影响通行或存在安全隐患，但未明确高度超标，选择：\"现场材料堆放不合理\"。\n"
                "（8）如描述涉及现场环境脏乱、垃圾堆积、管理混乱等问题，选择：\"现场脏乱\"。\n"
                "（9）如描述涉及施工现场存在积水，可能引发滑倒、触电等风险，选择：\"现场积水\"。\n"
                "（10）如描述涉及电线、电缆直接拖地、浸泡在水中或存在明显触电风险，选择：\"电线拖地、泡水\"。\n"
                "（11）如描述涉及施工现场随意吸烟、明火作业或可见烟雾，且未明确为规范动火作业，选择：\"随意吸烟、明火、烟雾\"。\n"
                "（12）如描述涉及动火作业，且邻近易燃或易爆物品，存在重大火灾或爆炸风险，选择：\"动火作业邻近易燃、易爆物品\"。\n"
                "（13）如以上所有风险类型均不符合，选择：\"无风险\"。\n\n"
                "【输出要求（非常重要）】：\n"
                "你必须且只能输出一个 JSON 对象，格式严格如下：\n"
                "{\"风险类型\": \"<枚举中的完整原文>\"}\n\n"
                "不得输出任何解释、理由、多余文本、标点或 Markdown 标记。\n\n"
                f"【风险描述】：{risk['风险描述']}"
            )
            step2_output = get_result_local(prompt=prompt)
            risk2 = extract_assistant_json(step2_output)
    except Exception as e:
        print(f"[WARN] Step2 失败，自动降级为无风险: {e}")
        step2_output = ""
        current_time = datetime.datetime.now()
        file_name = current_time.strftime('%Y%m%d_%H%M%S')
        save_json_to_results({"fjaf":"fjak"}, file_name)
        risk2 = _no_risk_result()

    # ========== 规范化风险类型 ==========
    class_risk = [
        "未带安全带/安全绳/生命线",
        "未带安全帽",
        "临边、洞口防护不足",
        "现场脏乱",
        "现场材料堆放不合理",
        "现场积水",
        "电线拖地、泡水",
        "随意吸烟、明火、烟雾",
        "动火作业邻近易燃、易爆物品",
        "材料堆放过高",
        "脚手架未设置斜撑",
        "基坑周边未设置防护杆",
        "无风险"
    ]

    risk_real = "无风险"
    for cla in class_risk:
        if cla in risk2.get("风险类型", ""):
            risk_real = cla
            break

    # ========== 保存完整日志 ==========
    save_data = {
        "firstdangerType": risk2.get("风险类型", "无风险"),
        "dangerType": risk_real,
        "dangerContent": risk.get("风险描述", "未发现明显施工安全风险"),
        "step1_output": step1_output,
        "step1_extract": risk,
        "step2_output": step2_output,
        "step2_extract": risk2,
        "image_path": image_path
    }

    current_time = datetime.datetime.now()
    file_name = current_time.strftime('%Y%m%d_%H%M%S')
    save_json_to_results(save_data, file_name)

    return {
        "dangerType": risk_real,
        "dangerContent": risk.get("风险描述", "未发现明显施工安全风险")
    }


if __name__ == "__main__":
    # result = get_result_api("aaa")
    # # print(result.json()["choices"][0]["message"]["content"])
    # output_dir = os.path.join(os.getcwd(), "save_corrected_picture", get_name())
    # print(output_dir)
    # print(os.walk(output_dir))

    # test llm api
    result = run("/home/synloop/risk/image/37282_首层砌体工程操作平台存在架上架现象.jpg")
    # print(result)

    # test extract_direction
    # print(extract_direction("/home/synloop/risk/save_corrected_picture/20260317_115544/face_right.jpg"))