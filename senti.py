import os
import nltk
import pandas as pd
import time
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from openai import OpenAI

# 下载NLTK资源（首次运行需要）
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# API配置
client = OpenAI(
    api_key="",
    base_url="",
)
model_name = "d"

input_dir = r""
output_dir = os.path.join(input_dir, "")
os.makedirs(output_dir, exist_ok=True)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None

def get_swn_score(word, pos=None):
    try:
        synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
        scores = []
        for syn in synsets:
            try:
                swn_syn = swn.senti_synset(syn.name())
                score = swn_syn.pos_score() - swn_syn.neg_score()
                scores.append(score)
            except:
                continue
        return sum(scores) / len(scores) if scores else 0.0
    except:
        return 0.0

def calc_sentence_swn_scores(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    scores = []
    word_scores = []
    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        score = get_swn_score(word, wn_pos)
        scores.append(score)
        word_scores.append(f"{word}({score:.2f})")
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, word_scores

def build_prompt(sentence, word_sense, definition, avg_score, word_scores):
    sentiment_info = ", ".join(word_scores)
    sentiment_summary = f"Sentence average SentiWordNet score: {avg_score:.3f}"
    prompt = (
        f"You are a sentiment analysis assistant. The sentence below contains a metaphor.\n\n"
        f"Sentence: {sentence}\n"
        f"Literal meaning: {word_sense}\n"
        f"Metaphorical meaning in context: {definition}\n"
        f"{sentiment_summary}\n"
        f"Word-level sentiment scores: {sentiment_info}\n\n"
        f"Please analyze the metaphorical usage of the sentence and determine whether its emotional polarity is Positive or Negative.\n"
        f"Only output one word: Positive or Negative."
    )
    return prompt

def query_qwen(prompt, retries=3, delay=5):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            result = res.choices[0].message.content.strip().lower()
            if "positive" in result:
                return "正面"
            elif "negative" in result:
                return "负面"
        except Exception as e:
            print(f"[错误] 尝试第 {attempt+1} 次失败：{e}")
            time.sleep(delay)
    return "失败"

# 批量处理所有tsv文件，每个只处理前1000行
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(".tsv"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        print(f"\n🔍 正在处理文件：{file_name}")

        df = pd.read_csv(input_path, sep="\t", encoding="utf-8-sig")
        df = df.head(1000)  # 只处理前1000行

        df["情感得分"] = None

        for idx, row in df.iterrows():
            sentence = str(row["filled_sentence"])
            word_sense = str(row.get("word_sense", ""))
            definition = str(row.get("definition", ""))

            avg_score, word_scores = calc_sentence_swn_scores(sentence)
            prompt = build_prompt(sentence, word_sense, definition, avg_score, word_scores)
            qwen_class = query_qwen(prompt)

            if qwen_class == "正面":
                label = 1
            elif qwen_class == "负面":
                label = 0
            else:
                label = -1  # 失败或无结果

            df.at[idx, "情感得分"] = label
            print(f"[{idx}] 句子隐喻情感: {qwen_class} => {label}")

        df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
        print(f"✅ 文件处理完成，结果已保存至：{output_path}")

print("\n🎉 所有文件处理完成！")

