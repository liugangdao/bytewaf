import tiktoken
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from openvino import convert_model
import openvino as ov
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class WAFDataset(Dataset):
    def __init__(self, samples, labels, max_len=64):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.samples = samples
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print( self.samples[idx])
        if not isinstance(self.samples[idx], str):
            self.samples[idx] = str(self.samples[idx])
            print(self.labels[idx], self.samples[idx])
        tokens = self.tokenizer.encode(self.samples[idx])
        tokens = tokens[: self.max_len] + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.float
        )


class ByteCNN(nn.Module):
    def __init__(self, vocab_size=100_261, max_len=64, embed_dim=128, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * max_len // 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Infer:
    def __init__(self, model_path):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def predict(self, text, max_len):

        tokens = self.tokenizer.encode(text)
        tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
        input_tensor = torch.tensor([tokens], dtype=torch.long).to("cuda:0")
        with torch.no_grad():
            score = self.model(input_tensor).item()
        return score, "攻击 🚨" if score > 0.5 else "正常 ✅"


def train_model(
    samples,
    labels,
    batch_size=4,
    vocab_size=100_261,
    max_len=64,
    embed_dim=128,
    lr=1e-3,
    epochs=10,
    save_path="waf_model.pth",
    test_size=0.1,
    patience=5,
    device="cuda:0",
):
    """
    训练 ByteBPE + CNN 模型进行 WAF 攻击检测

    参数:
        samples: List[str]，输入字符串（如 payloads）
        labels: List[int]，0 或 1（正常 / 攻击）
        batch_size: 批大小
        max_len: Token 最大长度
        embed_dim: Embedding 维度
        lr: 学习率
        epochs: 训练轮数
        save_path: 模型保存路径
    """

    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, test_size=test_size, stratify=labels, random_state=42
    )

    train_dataset = WAFDataset(X_train, y_train, max_len=max_len)
    val_dataset = WAFDataset(X_val, y_val, max_len=max_len)

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ByteCNN(vocab_size=vocab_size, max_len=max_len, embed_dim=embed_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    patience_counter = 0
    # best_model_state = None
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader_train):
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze()
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze()
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = (out > 0.5).int()
                correct += (preds == y.int()).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(
            f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss/len(loader_train):.4f} | Val Loss: {val_loss/len(loader_val):.4f} | Val Acc: {val_acc:.4f}"
        )

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_scripted = torch.jit.script(model)  # Export to TorchScript
            model_scripted.save(f"{save_path}/model.pt")  # Save
            ov_model = convert_model(
                input_model=f"{save_path}/model.pt",
                example_input=torch.rand(1, max_len),
            )
            ov.save_model(ov_model, output_model=f"{save_path}/model.xml")
            print(f"✅ 模型已保存到 {save_path}")
            # print("🔖 验证集准确率提升，保存当前模型")
        else:
            patience_counter += 1
            print(f"⏸️ 验证准确率未提升，patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 早停触发，停止训练")
                break


# 示例使用
if __name__ == "__main__":
    samples = [
        "id=1 OR 1=1",  # SQLi
        "<script>alert('x')</script>",  # XSS
        "page=home",  # 正常请求
        "user=admin&pass=1234",  # 正常请求
    ]
    labels = [1, 1, 0, 0]  # 1=攻击，0=正常
    # 从csv中获取数据
    pathxss = "/home/zhiwei/workspace/tadk-auto/.cache_xss/training_data.csv"
    pathsql = "/home/zhiwei/workspace/tadk-auto/.cache/training_data.csv"
    import pandas as pd

    dfxss = pd.read_csv(pathxss)
    dfsql = pd.read_csv(pathsql)
    print(dfxss.head())
    print(dfsql.head())

    # 合并dfxss和dfsql
    # 合并两个 DataFrame
    df_merged = pd.concat([dfxss, dfsql], ignore_index=True)

    # 根据 sample 列去重
    df_deduplicated = df_merged.drop_duplicates(subset="sample")
    df_deduplicated = df_merged.dropna(subset=["sample"])  # 去除 sample 是 NaN 的行
    df_deduplicated = df_deduplicated[
        df_deduplicated["sample"].str.strip() != ""
    ]  # 去除空字符串行

    df_deduplicated = df_deduplicated[
        df_deduplicated["sample"] != None
    ]  # 去除 sample 是 None 的行

    # 标签映射
    label_map = {"xss": 1, "sql": 1, "normal": 0}
    df_deduplicated["label"] = df_deduplicated["label"].map(label_map)

    # 保存去重后的数据df,只保存 sample和label两列
    df_deduplicated = df_deduplicated[["sample", "label"]]
    df_deduplicated.to_csv(
        "data/training_data_deduplicated.csv",
        index=False,
    )

    df_normal = pd.read_csv("data/rich_keywords_non_attack.csv")
    df_deduplicated = pd.concat(
        [
            df_normal,
            df_deduplicated,
        ],
        ignore_index=True,
    )
    # 提取样本和标签
    df_deduplicated = df_deduplicated[
        df_deduplicated["sample"] != None
    ]  # 去除 sample 是 None 的行
    df_deduplicated = df_deduplicated[
        df_deduplicated["sample"].str.strip() != ""
    ]  # 去除空字符串行
    df_deduplicated = df_deduplicated.drop_duplicates(subset="sample")
    print(df_deduplicated.head())

    texts = df_deduplicated["sample"].tolist()
    labels = df_deduplicated["label"].tolist()
    # 查看结果
    print(df_deduplicated)

    model_path = f"model_save"
    device = "cuda:0"
    max_len = 100
    train_model(
        texts, labels, batch_size=256, max_len=max_len, epochs=100, save_path=model_path
    )

    infer = Infer(f"{model_path}/model.pt")
    for text in samples:
        print(infer.predict(text, max_len=max_len))
