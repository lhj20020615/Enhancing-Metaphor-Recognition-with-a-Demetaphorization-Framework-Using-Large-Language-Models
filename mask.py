import pandas as pd

# 读取原始TSV文件
file_path = r""
df = pd.read_csv(file_path, sep='\t')

# 替换 sentence 中的目标词为 【】
def mask_target(row):
    try:
        words = row['sentence'].split()
        idx = int(row['w_index'])
        if 0 <= idx < len(words):
            words[idx] = '【】'
            return ' '.join(words)
        else:
            return row['sentence']
    except:
        return row['sentence']

# 仅替换 sentence 列
df['sentence'] = df.apply(mask_target, axis=1)

# 保存结果
output_path = file_path.replace(".tsv", "_masked.tsv")
df.to_csv(output_path, sep='\t', index=False)

print(f"句子处理完成，文件已保存为：{output_path}")