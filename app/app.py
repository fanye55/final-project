import os
import tarfile
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# -----------------------------
# 全局常量
# -----------------------------
DATA_URL = "https://dagshub.com/fanye55/final-project/raw/master/feature/dataset.tar.gz"
DATA_DIR = "dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = DEVICE  # ✅ pytest 要求存在

# -----------------------------
# 下载和解压 dataset
# -----------------------------
def download_dataset():
    if os.path.exists(DATA_DIR):
        print("✅ Dataset already exists")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = "dataset.tar.gz"

    print("📥 Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, tar_path)

    print("📦 Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()

    os.remove(tar_path)
    print("✅ Dataset ready")


# -----------------------------
# Dataset Class
# -----------------------------
class IrisTxtDataset(Dataset):
    def __init__(self, split):
        self.features = []
        self.labels = []

        feature_dir = os.path.join(DATA_DIR, split, "feature")
        label_dir = os.path.join(DATA_DIR, split, "label")

        feature_files = sorted(os.listdir(feature_dir))
        label_files = sorted(os.listdir(label_dir))

        for f, l in zip(feature_files, label_files):
            f_path = os.path.join(feature_dir, f)
            l_path = os.path.join(label_dir, l)

            feat = np.loadtxt(f_path)
            label = int(np.loadtxt(l_path))

            self.features.append(feat)
            self.labels.append(label)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx])
        )


# -----------------------------
# CNN Model
# -----------------------------
class CNN_Model(nn.Module):  # ✅ pytest 需要该类存在且可导入
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # ✅ 三分类
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)


# -----------------------------
# 初始化数据
# -----------------------------
download_dataset()

train_dataset = IrisTxtDataset("train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # ✅ pytest 需要 train_loader

# -----------------------------
# 评估函数
# -----------------------------
def evaluate_model(model, data_loader, desc="Evaluate"):  # ✅ pytest 需要 evaluate_model
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
    acc = accuracy_score(trues, preds)
    print(f"{desc} Accuracy: {acc:.4f}")
    return acc


# -----------------------------
# 训练入口（手动执行）
# -----------------------------
if __name__ == "__main__":
    model = CNN_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Start Training...")
    for epoch in range(3):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        evaluate_model(model, train_loader, desc=f"Epoch {epoch+1}")
    print("✅ Training Done")
