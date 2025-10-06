import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# ================= 配置参数 =================
BASE_DIR = ""
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRAIN_REPEAT = 10  # 每个训练集文件夹跑10轮
TRAIN_FILE_NAME = "train.tsv"
VAL_PATH = os.path.join(BASE_DIR, "v.tsv")
TEST_PATH = os.path.join(BASE_DIR, "t.tsv")
MODEL_NAME = ""
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_ACCUM_STEPS = 1  # 梯度累积步数，可根据显存调整

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# ================= 数据集 =================
class NounMetaphorDataset(Dataset):
    """
    数据集类：用于目标词名词性隐喻分类
    """
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = pd.read_csv(file_path, sep="\t", encoding='utf-8')

        # 过滤掉无效行
        self.data = self.data[
            self.data["sentence"].notna() &
            (self.data["sentence"].astype(str).str.strip() != "") &
            self.data["label"].notna() &
            self.data["w_index"].notna()
        ]

        # 转成整数
        self.data["label"] = self.data["label"].astype(int)
        self.data["w_index"] = self.data["w_index"].astype(int)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = str(row["sentence"])
        label = row["label"]
        w_index = row["w_index"]

        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "w_index": w_index  # 目标词索引
        }

# ================= BERT分类模型 =================
class BertForTargetMetaphorClassification(nn.Module):
    """
    BERT 二分类模型：使用 [CLS] + 目标词向量 拼接判断目标词是否为名词性隐喻
    """
    def __init__(self):
        super(BertForTargetMetaphorClassification, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, 2)  # 拼接后维度变成 2*hidden

    def forward(self, input_ids, attention_mask, w_index):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state       # [batch, seq_len, hidden]
        cls_embeddings = hidden_states[:, 0, :]         # [CLS] 向量 [batch, hidden]

        # 根据目标词索引提取对应 token 的向量
        batch_size = input_ids.size(0)
        target_embeddings = []
        for i in range(batch_size):
            idx = min(w_index[i], hidden_states.size(1)-1)  # 防止越界
            target_embeddings.append(hidden_states[i, idx, :])
        target_embeddings = torch.stack(target_embeddings)  # [batch, hidden]

        # 拼接 [CLS] + 目标词向量
        combined = torch.cat([cls_embeddings, target_embeddings], dim=1)  # [batch, hidden*2]

        logits = self.classifier(combined)
        return logits

# ================= 单次训练 + 测试 =================
def train_one_run(train_path, run_idx=1, output_folder=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🚀 第 {run_idx} 次训练开始前，已清理显存。")
        print(f"已分配: {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
              f"已保留: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    # 数据加载
    train_dataset = NounMetaphorDataset(train_path, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = NounMetaphorDataset(VAL_PATH, tokenizer, max_len=MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    global test_loader
    if 'test_loader' not in globals():
        test_dataset = NounMetaphorDataset(TEST_PATH, tokenizer, max_len=MAX_LEN)
        globals()['test_loader'] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型和优化器
    model = BertForTargetMetaphorClassification().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    best_train_acc = 0.0
    best_val_acc = 0.0
    best_epoch = 0  # 训练集最高准确率轮次
    best_val_epoch = 0  # 验证集最高准确率轮次（测试集使用的轮次）
    consecutive_fail = 0
    total_fail = 0
    val_fail = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            w_index = batch["w_index"]

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask, w_index)
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS, acc=correct / total)

        train_acc = correct / total
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_epoch = epoch + 1
            consecutive_fail = 0
            print(f"训练集准确率提升! 新最高: {best_train_acc:.4f}, 最佳训练轮: {best_epoch}")
        else:
            consecutive_fail += 1
            total_fail += 1
            print(f"未超过最高 {best_train_acc:.4f}, 连续失败 {consecutive_fail}, 总失败 {total_fail}")

        # 验证阶段
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                w_index = batch["w_index"]
                logits = model(input_ids, attention_mask, w_index)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"验证集准确率: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            val_fail = 0
            best_model_state = model.state_dict()
            print(f"验证集准确率提升! 新最高: {best_val_acc:.4f}, 最佳验证轮: {best_val_epoch}")
        else:
            val_fail += 1
            print(f"验证集未提升，连续 {val_fail} 次")

        # 提前停止
        if consecutive_fail >= 3 or total_fail >= 5 or val_fail >= 5:
            print(f"提前停止训练 (连续失败或验证集未提升)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 测试阶段
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            w_index = batch["w_index"]
            logits = model(input_ids, attention_mask, w_index)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    pre, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    precision_array, recall_array, f1_array, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0,1], average=None
    )
    pre_0, pre_1 = precision_array
    rec_0, rec_1 = recall_array
    f1_0, f1_1 = f1_array

    metrics = {
        "Accuracy": acc,
        "Macro_Precision": pre,
        "Macro_Recall": rec,
        "Macro_F1": f1,
        "Precision_0": pre_0,
        "Recall_0": rec_0,
        "F1_0": f1_0,
        "Precision_1": pre_1,
        "Recall_1": rec_1,
        "F1_1": f1_1,
        "Best_Train_Epoch": best_epoch,
        "Best_Val_Acc": best_val_acc,
        "Best_Val_Epoch": best_val_epoch
    }

    if output_folder:
        run_file = os.path.join(output_folder, f"run_{run_idx}.txt")
        os.makedirs(os.path.dirname(run_file), exist_ok=True)
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(f"训练集 {os.path.basename(train_path)} 第 {run_idx} 次名词性隐喻分类测试结果:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    return metrics

# ================= 每个训练集跑多次并保存 =================
def train_folder_average(folder_path, output_folder, repeat=TRAIN_REPEAT):
    os.makedirs(output_folder, exist_ok=True)
    metrics_list = []
    for i in range(repeat):
        print(f"\n=== 第 {i+1}/{repeat} 次训练 ===")
        metrics = train_one_run(folder_path, run_idx=i+1, output_folder=output_folder)
        metrics_list.append(metrics)

    avg_metrics = {k: sum(m[k] for m in metrics_list) / repeat for k in metrics_list[0] if k not in ["Best_Train_Epoch", "Best_Val_Epoch"]}
    avg_best_val_epoch = sum(m["Best_Val_Epoch"] for m in metrics_list) / repeat
    avg_metrics["Avg_Best_Val_Epoch"] = avg_best_val_epoch

    final_path = os.path.join(output_folder, "final_results.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(f"训练集 {os.path.basename(folder_path)} 平均名词性隐喻分类结果 (基于 {repeat} 次训练):\n")
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"\n训练集 {os.path.basename(folder_path)} 平均结果:\n{avg_metrics}")
    return avg_metrics

# ================= 主程序 =================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_results = []

    # 匹配训练集文件夹，大小写兼容
    subfolders = sorted([f for f in os.listdir(BASE_DIR) if f.lower().startswith("train")])

    for folder in subfolders:
        train_file = os.path.join(BASE_DIR, folder, TRAIN_FILE_NAME)
        output_folder = os.path.join(RESULTS_DIR, folder)

        if os.path.exists(train_file):
            print(f"\n=== 在训练集 {train_file} 上进行名词性隐喻分类训练 {TRAIN_REPEAT} 次并求平均 ===")
            avg_metrics = train_folder_average(train_file, output_folder, repeat=TRAIN_REPEAT)
            avg_metrics["folder"] = folder
            summary_results.append(avg_metrics)
        else:
            print(f"⚠️ {train_file} 不存在，跳过。")

    # 汇总所有训练集结果
    if summary_results:
        df = pd.DataFrame(summary_results)
        summary_csv = os.path.join(RESULTS_DIR, "summary_results.csv")
        df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ 所有训练集名词性隐喻分类汇总结果已保存到 {summary_csv}")
