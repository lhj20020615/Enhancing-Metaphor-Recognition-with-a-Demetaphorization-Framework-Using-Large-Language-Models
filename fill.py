import pandas as pd
import time
from openai import OpenAI

# ================= 配置 =================
client = OpenAI(
    api_key="sk-355ce744406f4abe8326ad52316eeae1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

input_path = r"C:\Users\lhj20200615\PycharmProjects\biaozhu\allen1_target_domain_tenor_filtered_masked.tsv"
output_path = input_path.replace(".tsv", "_with_literal_李浩嘉1_preview.tsv")

df = pd.read_csv(input_path, sep="\t")

# 添加新列
df["associated_literal_fills"] = ""
for n in range(1, 6):
    df[f"filled_sentence_{n}"] = ""
df["fill_word_index_1"] = -1  # 仅记录第一个词的索引

model_name = "qwen-plus"


# ================= 填词函数 =================
def generate_literal_fills(sentence, tenor, num_options=5):
    prompt = f"""The following sentence contains a **noun metaphor**. Fill in the blank 【】 with **{num_options} literal nouns** 
based on the given tenor. The words should have no metaphorical meaning and must be used in their original, literal sense.

Sentence (with blank): {sentence}
Tenor (metaphorical noun): {tenor}

Just return {num_options} literal nouns, separated by commas, without any explanations."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a linguistic expert specializing in literal word substitution."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        fills = [w.strip() for w in response.choices[0].message.content.strip().split(",")]
        while len(fills) < num_options:
            fills.append("")
        return fills
    except Exception as e:
        print(f"API call failed: {e}")
        return [""] * num_options


# ================= 主循环（仅前两行） =================
for i, row in df.head(1687).iterrows():  # 仅取前两行
    sentence = str(row["sentence"])
    tenor = str(row.get("tenor", ""))

    fills = generate_literal_fills(sentence, tenor, num_options=5)
    df.at[i, "associated_literal_fills"] = ", ".join(fills)

    # 生成每个候选词对应的句子
    for n, word in enumerate(fills, start=1):
        if word and "【】" in sentence:
            filled_sentence = sentence.replace("【】", word)
        else:
            filled_sentence = sentence
        df.at[i, f"filled_sentence_{n}"] = filled_sentence

    # 查找第一个词在句子中的索引
    try:
        words = df.at[i, "filled_sentence_1"].split()
        index = words.index(fills[0])
        df.at[i, "fill_word_index_1"] = index
    except (ValueError, IndexError):
        df.at[i, "fill_word_index_1"] = -1

    print(f"Row {i + 1} done → Words: {fills} | Index of first: {df.at[i, 'fill_word_index_1']}")
    time.sleep(1.2)

# 保存结果
df.to_csv(output_path, sep="\t", index=False)
print(f"\n✅ 预览处理完成，结果保存在：{output_path}")
