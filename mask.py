import pandas as pd
import re

# 加载 TSV 文件
input_path = r"C:\Users\lhj20\PycharmProjects\pythonProjectlw\shiyanzhu\gpt-4o.tsv"
df = pd.read_csv(input_path, sep="\t", encoding="utf-8")

# 替换句子中的【】内容
def replace_brackets(sentence, fill_word):
    # 确保 sentence 和 fill_word 是字符串类型
    sentence = str(sentence) if sentence is not None else ""
    fill_word = str(fill_word) if fill_word is not None else ""
    return re.sub(r"【.*?】", fill_word, sentence)

# 应用替换操作
df["filled_sentence"] = df.apply(lambda row: replace_brackets(row["sentence"], row["associated_literal_fill"]), axis=1)

# 保存为新文件
output_path = input_path.replace(".tsv", "_replaced.tsv")
df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

print(f"替换完成，已保存到：{output_path}")
