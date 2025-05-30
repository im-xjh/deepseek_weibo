import json
import csv

# 文件路径
input_file = "weibo_deepseek.jsonl"
output_csv_file = "cleaned_data.csv"
output_jsonl_file = "cleaned_data.jsonl"


# 定义变量统计原始和清洗后的数据量
original_count = 0
final_count = 0

data = []
unique_ids = set()
unique_contents = set()

# 打开JSONL文件并清洗数据
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        original_count += 1
        record = json.loads(line)

        # 保留content中包含"deepseek"的行
        if "deepseek" not in record["content"].lower():
            continue

        # 提取url中的id，去除?及其后面的内容
        url = record.get("url", "")
        post_id = url.split("/")[-1].split("?")[0] if url else ""

        # 修改字段名
        record["id"] = post_id
        record["user"] = record.pop("user_name", "")
        record["time"] = record.pop("post_time", "")
        record["ip"] = record.pop("ip_text", "").replace("发布于 ", "")
        record["url"] = url

        # 将url放到末尾，id放到首位
        cleaned_record = {
            "id": record["id"],
            "user": record["user"],
            "time": record["time"],
            "ip": record["ip"],
            "content": record["content"],
            "forward": record["forward"],
            "comment": record["comment"],
            "like": record["like"],
            "url": record["url"],
        }

        # 去重处理
        if cleaned_record["id"] not in unique_ids and cleaned_record["content"] not in unique_contents:
            unique_ids.add(cleaned_record["id"])
            unique_contents.add(cleaned_record["content"])
            data.append(cleaned_record)

# 统计清洗后的数据量
final_count = len(data)

# 将清洗后的数据写入CSV文件
with open(output_csv_file, "w", encoding="utf-8", newline="") as csvfile:
    fieldnames = ["id", "user", "time", "ip", "content", "forward", "comment", "like", "url"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    # 写入数据
    writer.writerows(data)

# 将清洗后的数据写入JSONL文件
with open(output_jsonl_file, "w", encoding="utf-8") as jsonlfile:
    for record in data:
        jsonlfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"原始数据量: {original_count}")
print(f"清洗去重后的数据量: {final_count}")
