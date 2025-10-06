import os
import pandas as pd
import time
from openai import OpenAI
from nltk.corpus import wordnet as wn
import nltk

# ================= 配置 =================
nltk.download('wordnet')
nltk.download('omw-1.4')

client_qwen = OpenAI(
    api_key="",  # 替换为你自己的 Key
    base_url="",
)
model_qwen = "deepseek-r1"

input_path = r""
output_path = input_path.replace(".tsv", "_processed.tsv")

# ================= 读取数据并支持断点续传 =================
if os.path.exists(output_path):
    df = pd.read_csv(output_path, sep="\t", encoding="utf-8")
    last_done = df[df["ProcessedSentence"].notna()].index.max()
    start_idx = last_done + 1 if pd.notna(last_done) else 0
    print(f"🔄 断点续传，从第 {start_idx+1} 行继续")
else:
    df = pd.read_csv(input_path, sep="\t", encoding="gbk")
    df["ProcessedSentence"] = ""
    df["TargetIndex"] = df["w_index"]
    df["NewMetaphorFlag"] = 0
    df["ReplacementWord"] = ""
    start_idx = 0
    print("🚀 从头开始处理")

# ================= 工具函数 =================
def get_wordnet_definitions(word):
    """获取目标词的WordNet释义作为参考"""
    synsets = wn.synsets(word, pos=wn.NOUN)
    defs = [s.definition() for s in synsets]
    return "; ".join(defs) if defs else "No definition"

def check_nominal_metaphor(sentence, target_word):
    """结合整个句子判断目标词是否为名词性隐喻"""
    word_def = get_wordnet_definitions(target_word)
    prompt = f"""
You are a linguistic expert in metaphor detection. 
Check if the target word in the sentence is used as a nominal metaphor.
Focus on the context of the sentence; use the WordNet definition only as reference.

Sentence: {sentence}
Target word: {target_word}
Word definition (reference): {word_def}

Answer only with "true" or "false".
"""
    try:
        response = client_qwen.chat.completions.create(
            model=model_qwen,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return "true" in response.choices[0].message.content.lower()
    except Exception as e:
        print(f"Metaphor check error: {e}")
        return False

def replace_if_metaphor(sentence, target_index):
    """如果目标词是名词性隐喻，则替换为非隐喻词；否则保持原词"""
    words = sentence.split()
    if target_index < 0 or target_index >= len(words):
        # 如果索引超出范围 → 取最后一个词
        target_index = len(words) - 1

    target_word = words[target_index]
    is_metaphor = check_nominal_metaphor(sentence, target_word)

    replacement = target_word
    new_flag = 0

    if is_metaphor:
        # 使用模型直接选择一个非隐喻替换词（不依赖WordNet列表）
        prompt_replace = f"""
The target word "{target_word}" in the sentence "{sentence}" is used metaphorically.
Provide a single natural, non-metaphorical replacement word.
Output only the replacement word.
"""
        try:
            response = client_qwen.chat.completions.create(
                model=model_qwen,
                messages=[{"role": "user", "content": prompt_replace}],
                temperature=0.3
            )
            replacement_candidate = response.choices[0].message.content.strip()
            if replacement_candidate and replacement_candidate.lower() != target_word.lower():
                replacement = replacement_candidate
        except Exception as e:
            print(f"Replacement error: {e}")

        words[target_index] = replacement
        new_flag = int(check_nominal_metaphor(" ".join(words), replacement))

    new_sentence = " ".join(words)
    return new_sentence, replacement, new_flag, target_index

# ================= 遍历处理 =================
for i in range(start_idx, len(df)):
    row = df.iloc[i]
    sentence = str(row["sentence"])
    target_index = int(row["w_index"])
    new_sentence, replacement_word, new_flag, actual_index = replace_if_metaphor(sentence, target_index)

    df.at[i, "ProcessedSentence"] = new_sentence
    df.at[i, "TargetIndex"] = actual_index
    df.at[i, "NewMetaphorFlag"] = new_flag
    df.at[i, "ReplacementWord"] = replacement_word

    print(f"Row {i+1}: original='{sentence.split()[actual_index]}', replacement='{replacement_word}', new_flag={new_flag}")

    # ================= 每行处理完后保存 =================
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    time.sleep(0.1)  # 可根据需要调整

print(f"\n✅ Processing complete. File saved to: {output_path}")
