import pandas as pd

# 1. 读取 JSONL 数据文件
input_file = "/Users/jhx/Documents/Code/deep/原始数据.jsonl"
df = pd.read_json(input_file, lines=True)

# 2. 筛选出符合要求的数据：
#    (1) comments_count 不为 0
#    (2) is_retweet 为 False（即原创微博）
filtered_df = df[(df["comments_count"] > 5) & (df["is_retweet"] == False)]

# 3. 另存为新的 JSONL 文件（也可以保存为 CSV 格式）
output_file = "筛选结果.jsonl"
filtered_df.to_json(output_file, orient="records", lines=True, force_ascii=False)

print(f"筛选完成，结果已保存至 {output_file}")