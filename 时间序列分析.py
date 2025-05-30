import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime

# 设置字体路径
FONT_PATH = "资源/msyh.ttf"
font = FontProperties(fname=FONT_PATH)

def load_jsonl_data(file_path):
    """加载JSONL文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def parse_custom_time(time_str):
    """解析自定义格式的时间字符串"""
    try:
        return datetime.strptime(time_str, '%y-%m-%d %H:%M')
    except ValueError:
        return None

def create_time_series_plot(df):
    """创建时间序列图"""
    # 解析时间字段
    df['datetime'] = df['time'].apply(parse_custom_time)
    
    # 移除解析失败的记录
    df = df.dropna(subset=['datetime'])
    
    # 设置时间范围（1.24-1.27）
    start_time = datetime(2025, 1, 24, 0, 0)
    end_time = datetime(2025, 1, 27, 23, 59)
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    df = df[mask]
    
    # 按小时统计频数（使用新的 'h' 参数）
    df['hour'] = df['datetime'].dt.floor('h')  # 将时间向下取整到小时
    hourly_counts = df.groupby('hour').size()
    
    # 创建完整的小时索引（使用新的 'h' 参数）
    idx = pd.date_range(start=start_time, end=end_time, freq='h')
    hourly_counts = hourly_counts.reindex(idx, fill_value=0)
    
    # 创建图表
    plt.figure(figsize=(15, 6))
    
    # 绘制时间序列图
    plt.plot(hourly_counts.index, hourly_counts.values, 
            color="#4C72B0", linewidth=2, marker='o', 
            markersize=4, markerfacecolor='white')
    
    # 设置标题和标签
    plt.title("时间序列分布图", fontsize=16, pad=12, fontproperties=font)
    plt.xlabel("时间", fontsize=12, fontproperties=font)
    plt.ylabel("频数", fontsize=12, fontproperties=font)
    
    # 设置网格
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 设置x轴刻度格式和旋转角度
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:00'))
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表到当前目录
    plt.savefig("分析结果/时间序列.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # 加载数据
    file_path = 'cleaned_data.jsonl'
    df = load_jsonl_data(file_path)
    
    # 生成时间序列图
    create_time_series_plot(df)
    
    # 打印一些统计信息
    print("数据时间范围：")
    print(f"开始时间：{df['datetime'].min()}")
    print(f"结束时间：{df['datetime'].max()}")
    print(f"总记录数：{len(df)}")

if __name__ == "__main__":
    main()