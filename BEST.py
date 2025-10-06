import os
import pandas as pd

# 修改成你的结果总目录
RESULTS_DIR = r"C:\Users\lhj20\Desktop\lastji\results"

def parse_run_file(file_path):
    """解析 run_x.txt，返回字典"""
    metrics = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                key = key.strip()
                try:
                    val = float(val.strip())
                except:
                    continue
                metrics[key] = val
    return metrics

def collect_results(results_dir):
    summary = []
    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.startswith("run_") and file.endswith(".txt"):
                run_path = os.path.join(folder_path, file)
                run_idx = file.split("_")[1].split(".")[0]  # 提取 run 序号
                metrics = parse_run_file(run_path)
                metrics["Train_Folder"] = folder
                metrics["Run_Index"] = int(run_idx)
                summary.append(metrics)

    return summary

if __name__ == "__main__":
    results = collect_results(RESULTS_DIR)
    if not results:
        print("⚠️ 没有找到任何 run_x.txt 文件")
        exit()

    # 保存完整结果
    df_all = pd.DataFrame(results)
    all_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    df_all.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存所有运行结果到 {all_path}")

    # 每个训练集选取 ACC 最高的 run
    best_rows = []
    for folder, group in df_all.groupby("Train_Folder"):
        best_idx = group["Accuracy"].idxmax()   # 找到 ACC 最高行
        best_rows.append(df_all.loc[best_idx])

    df_best_acc = pd.DataFrame(best_rows).reset_index(drop=True)

    best_path = os.path.join(RESULTS_DIR, "best_acc_per_trainset.csv")
    df_best_acc.to_csv(best_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存每个训练集 ACC 最佳的一次结果到 {best_path}")
