import json
import csv
import os
import requests
import logging
import sys
import time
from collections import deque

# 定义文件路径
input_file_path = 'cleaned_data.jsonl'
output_file_path = 'cleaned_data_sentiments.jsonl'
stats_file_path = '分析结果/情感分析统计.csv'
keys_file_path = '资源/keys.txt'  

# 设置日志记录
logging.basicConfig(
    filename='运行日志.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# 请求时间记录队列，用于控制 QPS
request_times = deque()

# 从外部文件读取API_KEY和SECRET_KEY
def load_keys(file_path):
    keys = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                keys[key] = value
    return keys.get('API_KEY', ''), keys.get('SECRET_KEY', '')

# 获取百度API Access Token
def get_access_token(api_key, secret_key):
    url = f'https://aip.baidubce.com/oauth/2.0/token'
    params = {
        'grant_type': 'client_credentials',
        'client_id': api_key,
        'client_secret': secret_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get('access_token')

# 进行情感分析
def sentiment_analysis(text, access_token):
    url = f"https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token={access_token}"
    headers = {'Content-Type': 'application/json'}
    data = {"text": text}  # 修改：使用"text"字段

    # 控制 QPS，不超过每秒 2 次请求
    while True:
        current_time = time.time()
        while request_times and current_time - request_times[0] >= 1:
            request_times.popleft()
        if len(request_times) < 4:
            break
        time.sleep(1 - (current_time - request_times[0]))

    try:
        response = requests.post(url, headers=headers, json=data)
        request_times.append(time.time())
        response.raise_for_status()
        result = response.json()
        item = result.get("items", [{}])[0]
        return item.get("sentiment"), item.get("confidence"), item.get("positive_prob"), item.get("negative_prob")
    except Exception as e:
        logging.error("情感分析请求失败: %s", e)
        return None, None, None, None

# 读取 JSONL 文件
def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                json_obj = json.loads(line.strip())
                if "content" not in json_obj:
                    logging.warning(f"第 {line_number} 行缺少 'content' 字段，跳过：{json_obj}")
                else:
                    data.append(json_obj)
            except json.JSONDecodeError as e:
                logging.error(f"第 {line_number} 行 JSON 格式错误，跳过：{line.strip()}")
    logging.info(f"成功读取 {len(data)} 条数据")
    return data

# 写入 JSONL 文件
def write_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

# 统计情感数据
def count_sentiments(data):
    sentiment_counts = {0: 0, 1: 0, 2: 0}
    for item in data:
        sentiment = item.get('sentiment')
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    sentiment_labels = {0: '负面', 1: '中性', 2: '正面'}
    stats = {sentiment_labels[k]: v for k, v in sentiment_counts.items()}
    return stats

# 写入统计结果到 CSV
def write_stats_to_csv(stats, file_path):
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['情感类型', '数量'])
        for sentiment, count in stats.items():
            writer.writerow([sentiment, count])

# 临时保存文件路径
progress_file_path = '/Users/jhx/Documents/Code/黑神话女性数据/0数据/sentiment_analysis_progress.jsonl'

# 读取已保存的进度
def load_progress(file_path):
    processed_data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                processed_data.append(json.loads(line.strip()))
    logging.info(f"已加载 {len(processed_data)} 条已处理数据")
    return processed_data

# 写入已处理数据到进度文件
def save_progress(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

# 修改情感分析函数以支持进度保存
def perform_sentiment_analysis(data, access_token, output_file_path, progress_file_path, report_interval=100):
    processed_data = load_progress(progress_file_path)
    processed_texts = {item['content'] for item in processed_data}  # 已处理文本集合
    total_items = len(data)

    for index, item in enumerate(data):
        if item['content'] in processed_texts:
            continue  # 跳过已处理的内容

        sentiment, confidence, positive_prob, negative_prob = sentiment_analysis(item['content'], access_token)
        item.update({
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': positive_prob,
            'negative_prob': negative_prob
        })
        save_progress([item], progress_file_path)  # 逐条保存进度

        if (index + 1) % report_interval == 0:
            logging.info(f"已完成 {index + 1}/{total_items} 条情感分析...")
            write_data(data[:index + 1], output_file_path)

    write_data(data, output_file_path)
    return data

# 修改主函数支持进度恢复
def main(input_path, output_path, stats_path, progress_path, keys_path):
    if os.path.exists(progress_path):
        os.remove(progress_path)
    
    
    api_key, secret_key = load_keys(keys_path)  # 从外部文件加载API_KEY和SECRET_KEY
    access_token = get_access_token(api_key, secret_key)
    logging.info("成功获取 Access Token")

    data = read_data(input_path)
    data = perform_sentiment_analysis(data, access_token, output_path, progress_path)
    logging.info("情感分析完成，结果已保存到文件")

    stats = count_sentiments(data)
    write_stats_to_csv(stats, stats_path)
    logging.info(f"情感统计结果已保存到 CSV 文件：{stats_path}")

if __name__ == "__main__":
    main(input_file_path, output_file_path, stats_file_path, progress_file_path, keys_file_path)