import pandas as pd

# 1. 读取JSONL数据文件
file_path = "/Users/jhx/Documents/Code/deep/原始数据.jsonl"
df = pd.read_json(file_path, lines=True)

# 2. 统计总帖子数量
total_count = len(df)

# 3. 统计原创微博（"is_retweet": false）的数量及比例
orig_df = df[df["is_retweet"] == False]
original_count = len(orig_df)
original_ratio = original_count / total_count if total_count > 0 else 0

# 4. 根据 _id 进行去重，统计去重后的准确数量
unique_df = df.drop_duplicates(subset=["_id"])
unique_count = len(unique_df)

# 5. 统计 reposts_count、comments_count、attitudes_count 均为 0 的帖子数量及比例
zero_interaction_df = df[
    (df["reposts_count"] == 0) &
    (df["comments_count"] == 0) &
    (df["attitudes_count"] == 0)
]
zero_interaction_count = len(zero_interaction_df)
zero_interaction_ratio = zero_interaction_count / total_count if total_count > 0 else 0

# 6. 整理统计结果，构建结果数据框
result_data = {
    "指标": [
        "总帖子数量",
        "原创微博数量",
        "原创微博比例",
        "去重后准确数量",
        "互动全为0的帖子数量",
        "互动全为0的帖子比例"
    ],
    "值": [
        total_count,
        original_count,
        round(original_ratio, 4),
        unique_count,
        zero_interaction_count,
        round(zero_interaction_ratio, 4)
    ]
}

result_df = pd.DataFrame(result_data)

# 7. 保存结果到CSV文件，确保表格结构清晰，数据完整
result_df.to_csv("统计结果.csv", index=False, encoding="utf-8-sig")

print("统计完成，结果已保存至 统计结果.csv")