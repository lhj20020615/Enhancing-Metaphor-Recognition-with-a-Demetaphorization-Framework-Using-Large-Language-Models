import pandas as pd
import time
from openai import OpenAI
from nltk.corpus import wordnet as wn  # WordNet

# ========== API 初始化 ==========
client_gpt4o = OpenAI(
    api_key="",
    base_url=""
)
model_gpt4o = ""

# ========== 输入输出路径 ==========
input_path = r""
output_path = input_path.replace(".tsv", "01.tsv")

# ========== 读取数据 ==========
df = pd.read_csv(input_path, sep="\t")
df = df.iloc[:].copy()
df["metaphor_flag"] = ""

# ========== 获取 WordNet 释义 ==========
def get_wordnet_definitions(word):
    synsets = wn.synsets(word)
    defs = [s.definition() for s in synsets]
    return "; ".join(defs) if defs else "No definition found"

# ========== 隐喻判断函数 ==========
def is_metaphor(sentence, word, index):
    # 获取 WordNet 定义
    definition = get_wordnet_definitions(word)

    prompt = f"""Determine whether the specified word in the following sentence is used metaphorically.

Sentence: {sentence}
Target word: {word}
Word sense(s): {definition}
Word position (index): {index}

Please respond with only "Yes" or "No", without any explanation."""

    try:
        response = client_gpt4o.chat.completions.create(
            model=model_gpt4o,
            messages=[
                {"role": "system", "content": "You are a linguistics expert in metaphor identification."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip().lower()
        if "yes" in answer:
            return 1
        elif "no" in answer:
            return 0
        else:
            return -1  # Cannot determine
    except Exception as e:
        print(f"API call failed: {e}")
        return -1

# ========== 遍历调用 ==========
for i, row in df.iterrows():
    sentence = str(row["sentence"])
    target_word = str(row["target"])
    word_index = int(row["w_index"])

    flag = is_metaphor(sentence, target_word, word_index)
    df.at[i, "metaphor_flag"] = flag

    print(f"Row {i+1} processed: metaphor_flag={flag}")
    time.sleep(1.2)  # 避免速率限制

# ========== 保存结果 ==========
df.to_csv(output_path, sep="\t", index=False)
print(f"\nProcessing complete. File saved to: {output_path}")
