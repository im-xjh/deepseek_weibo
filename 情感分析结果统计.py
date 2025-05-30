import os
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm

INPUT_FILE = "cleaned_data_sentiments.jsonl"        # 读取的文件
POS_LEXICON = "资源/positive_words.txt"                  # 积极情感词典
NEG_LEXICON = "资源/negative_words.txt"                  # 消极情感词典
CUSTOM_DICT = "资源/dic.txt"                             # 自定义词典
STOPWORDS_FILE = "资源/hit_stopwords.txt"                # 停用词
FONT_PATH = "资源/msyh.ttf"                              # 字体路径
OUTPUT_DIR = "分析结果"                          # 输出结果存放目录

# 设置 matplotlib 的字体属性
custom_font = fm.FontProperties(fname=FONT_PATH)
matplotlib.rcParams['font.sans-serif'] = [custom_font.get_name()]  # 使用加载的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成分布曲线图函数

def generate_distribution_chart(data, filename):
    """生成分布曲线图并保存"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, shade=True, color="#4C72B0", linewidth=2.5, alpha=0.6)

    plt.title("情绪概率密度分布 (0=消极, 1=积极)", fontsize=16, pad=12)
    plt.xlabel("积极概率值", fontsize=12)
    plt.ylabel("密度", fontsize=12)
    plt.grid(axis='both', linestyle=':', alpha=0.6)

    # 设置坐标轴范围
    plt.xlim(-0.02, 1.02)
    plt.ylim(bottom=0)

    # 设置刻度样式
    plt.xticks(np.linspace(0, 1, 6), fontsize=10)
    plt.yticks(fontsize=10)

    # 添加图例
    plt.legend(fontsize=10, frameon=True, facecolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # 1. 读取数据
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_json = json.loads(line)
            records.append(data_json)

    df = pd.DataFrame(records)

    # 2. 根据sentiment进行情感分析的统计, 并输出统计表格为CSV
    # sentiment: 0=消极, 1=中立, 2=积极
    sentiment_map = {0: '消极', 1: '中立', 2: '积极'}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['情感类型', '数量']
    sentiment_counts.to_csv(os.path.join(OUTPUT_DIR, '情感分析统计.csv'), index=False, encoding='utf-8-sig')

    # 3. 生成情感分布曲线图 (根据 positive_prob )
    positive_probs = df['positive_prob'].values
    generate_distribution_chart(positive_probs, '情感分布.png')

    # 4. 文本分词与形容词TF-IDF计算
    #    (1) 加载自定义词典
    if os.path.exists(CUSTOM_DICT):
        jieba.load_userdict(CUSTOM_DICT)

    #    (2) 加载停用词
    stopwords = set()
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as swf:
            for sw in swf:
                stopwords.add(sw.strip())

    #    (3) 加载积极和消极词表
    pos_words = set()
    if os.path.exists(POS_LEXICON):
        with open(POS_LEXICON, 'r', encoding='utf-8') as pf:
            for w in pf:
                pw = w.strip()
                if pw:
                    pos_words.add(pw)

    neg_words = set()
    if os.path.exists(NEG_LEXICON):
        with open(NEG_LEXICON, 'r', encoding='utf-8') as nf:
            for w in nf:
                nw = w.strip()
                if nw:
                    neg_words.add(nw)

    # 对content分词, 只保留形容词(a), 并过滤长度小于2的词
    all_contents = df['content'].fillna('').tolist()
    tokenized_texts = []

    for text in all_contents:
        words_with_pos = pseg.lcut(text)
        tokens_filtered = [word for (word, flag) in words_with_pos
                           if flag == 'a' and len(word) > 1 and word not in stopwords]
        tokenized_texts.append(' '.join(tokens_filtered))

    # 计算TF-IDF
    vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
    words = vectorizer.get_feature_names_out()

    tfidf_sums = tfidf_matrix.sum(axis=0)
    tfidf_sums = np.asarray(tfidf_sums).flatten()

    word_tfidf_dict = {w: s for w, s in zip(words, tfidf_sums)}

    # 分别挑出积极和消极词
    pos_tfidf = {}
    neg_tfidf = {}

    for w, score in word_tfidf_dict.items():
        if w in pos_words:
            pos_tfidf[w] = score
        elif w in neg_words:
            neg_tfidf[w] = score

    pos_df = pd.DataFrame(list(pos_tfidf.items()), columns=["词语", "TF-IDF总和"])
    pos_df.sort_values(by="TF-IDF总和", ascending=False, inplace=True)
    neg_df = pd.DataFrame(list(neg_tfidf.items()), columns=["词语", "TF-IDF总和"])
    neg_df.sort_values(by="TF-IDF总和", ascending=False, inplace=True)

    pos_df.to_csv(os.path.join(OUTPUT_DIR, "积极词_TFIDF.csv"), index=False, encoding='utf-8-sig')
    neg_df.to_csv(os.path.join(OUTPUT_DIR, "消极词_TFIDF.csv"), index=False, encoding='utf-8-sig')

    # 5. 制作词云图
    def generate_wordcloud(word_tfidf, out_file, title):
        if not word_tfidf:
            print(f"[警告] 没有可用词汇, 无法生成 {out_file} 词云图")
            return

        wc = WordCloud(
            font_path=FONT_PATH,  # 指定中文字体
            width=2000,
            height=1000,
            background_color="white",
            prefer_horizontal=1.0,  # 文字尽量横向排列
            margin=5  # 适当增加字与字的间隔
        )
        wc.generate_from_frequencies(word_tfidf)

        # 使用 Matplotlib 绘制并添加标题
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")  # 关闭坐标轴
        plt.title(title, fontproperties=custom_font, fontsize=15)  # 添加标题

        # 保存带有标题的词云图
        plt.savefig(os.path.join(OUTPUT_DIR, out_file), dpi=300, bbox_inches="tight")
        plt.close()

    # 生成词云
    generate_wordcloud(pos_tfidf, "积极词云.png", "积极词云图")
    generate_wordcloud(neg_tfidf, "消极词云.png", "消极词云图")

    print("情感分析处理完成, 结果已保存至:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

