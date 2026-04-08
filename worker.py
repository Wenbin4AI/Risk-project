import redis
import json
import requests
import time
import logging
import os
from utils import *

logging.basicConfig(level=logging.INFO)

r = redis.Redis(host='localhost', port=6379, db=0)
QUEUE_NAME = "image_tasks"

def process_image(image_url, data):
    try:
        image_type = data.get('image_type', '01')
        target_lang = data.get("lang")  # "ch", "en", "ja", "ar" 等
        results = []

        # 检测是否需要图片矫正
        if image_type == "02" or image_type == 2:
            output_dir = os.path.join(os.getcwd(), "save_corrected_picture", get_name())
            logging.info(f"检测到全景图，进行畸变矫正处理...")
            distortion_correction_six(image_url, output_dir, resolution=2048)
            # 模型结果
            logging.info(f"LLM处理...")

            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            image_files = []

            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        image_files.append(os.path.join(root, file).replace("\\", "/"))
            
            for item in image_files:
                start_time = time.time()
                result = run(str(item))
                direction = extract_direction(str(item))
                result['direction'] = direction
                logging.info(f"{item} 的处理结果为 {result}")
                if result["dangerType"] != "无风险":
                    result = convert_to_json(result, item, ".jpg", is_pano=True)
                    end_time = time.time()
                    run_time = end_time - start_time
                    # print(f"图片{str(item)}算法运行时间：{run_time}秒")
                    break
            if result["dangerType"] == "无风险":
                result = {'dangerType':'无风险', 'image': ""}
        else:
            result = run(str(image_url))
            result = convert_to_json(result, image_url, is_pano=False)


        final_result = [result]

        # 翻译为制定语言
        if target_lang in ["en", "ja", "ar"]:
            logging.info(f"检测到语言参数: {target_lang}，开始翻译...")
            final_result = [translate_entry(entry, target_lang) for entry in final_result]
        
        # 输出进行封装
        response_data = json.dumps(final_result, ensure_ascii=False)
        debug_result = []

        # 制作显示的检测数据
        for entry in final_result:
            debug_entry = entry.copy()
            if 'image' in debug_entry:
                debug_entry['image'] = f"<{type(debug_entry['image']).__name__}>"
            debug_result.append(debug_entry)

        logging.info(f"回调数据：{json.dumps(debug_result, ensure_ascii=False, indent=2)}")

        save_json_to_results(debug_result, get_name())

        return final_result

    except Exception as e:
        print(f"错误: {str(e)}")
    
def process_task(data):
    image_url = data["image_url"]
    callback_url = data["callback_url"]

    logging.info(f"Processing {image_url}")

    # ===== 在这里写你的算法 =====
    image_url = download_image(image_url)
    print(image_url)

    start_time = time.time()
    if image_url and callback_url:
        response_data = process_image(image_url, data)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info(f"The whole algorithm consume {run_time} seconds")

    # 回调
    # requests.post(callback_url, json=result)

    # 回调结果
    headers = {'Content-Type': 'application/json'}

    try:
        #发送POST请求,设置10秒超时
        if response_data[0]["image"] == "":
            response = requests.post(
                url=callback_url,
                headers=headers,
                json=[],#自动将字典转为JSoN字符串并设置Content-Type
                timeout=10
            )
        else:
            response = requests.post(
                url=callback_url,
                headers=headers,
                json=response_data,#自动将字典转为JSoN字符串并设置Content-Type
                timeout=10
            )
        #打印响应信息
        logging.info(f"响应状态码:{response.status_code}")
        logging.info(f"响应头: {response.headers}")
        logging.info(f"响应内容:{response.text}")

    except requests.exceptions.Timeout:
        logging.info("错误:请求超时")
    except requests.exceptions.ConnectionError:
        logging.info("错误:无法连接到服务器")
    except Exception as e:
        logging.info(f"其他错误:{str(e)}")

while True:
    try:
        # 阻塞式获取任务
        _, task = r.brpop(QUEUE_NAME)

        data = json.loads(task)

        process_task(data)

    except Exception as e:
        logging.error(e)
        time.sleep(1)