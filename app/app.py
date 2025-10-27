import os
import argparse
import urllib.request
import tarfile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== 配置部分 ==========
DATASET_URL = "https://dagshub.com/fanye55/final-project/raw/main/dataset.tar.gz"
DATASET_DIR = "./dataset"
DATASET_TAR = "./dataset.tar.gz"
RESULT_DIR = "./train-result"

os.makedirs(RESULT_DIR, exist_ok=True)


# ========== 下载进度条 ==========
class ProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        self.pbar.update(downloaded - self.pbar.n)


# ========== 下载与解压 ==========
def download_and_extract_dataset():
    if os.path.exists(DATASET_DIR):
        print("✅ Dataset already exists. Skipping download.")
        return

    print("🌍 Downloading dataset from DagsHub...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_TAR, ProgressBar())

    print("📦 Extracting dataset...")
    with tarfile.open(DATASET_TAR, "r:gz") as tar:
        tar.extractall()

    os.remove(DATASET_TAR)
    print("✅ Dataset ready!")


# ========== 自定义数据集 ==========
class FlowerDataset(Dataset):
    def __init__(self, root):
        self.features = []
        self.labels = []

        feature_dir = os.path.join(root, "feature")
        label_dir = os.path.join(root, "label")

        for file in sorted(os.listdir(feature_dir)):
            if file.endswith(".txt"):
                feature_path = os.path.join(feature_dir, file)
                label_path = os.path.join(label_dir, file.replace("feature", "label"))

                with open(feature_path, "r") as f:
                    feature = list(map(float, f.read().strip().split()))
                    self.features.append(feature)

                with open(label_path, "r") as f:
                    label = int(f.read().strip())
                    self.labels.append(label)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ========== 简单MLP分类模型 ==========
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 三分类
        )

    def forward(self, x):
        return self.net(x)


# ========== 训练函数 ==========
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ========== 测试 ==========
def evaluate(model, loader, criterion, device, split_name):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    print(f"✅ {split_name} | Loss: {total_loss / len(loader.dataset):.4f} | Acc: {correct / len(loader.dataset):.4f}")


# ========== 主入口 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.no_download:
        download_and_extract_dataset()

    train_ds = FlowerDataset(os.path.join(DATASET_DIR, "train"))
    val_ds = FlowerDataset(os.path.join(DATASET_DIR, "val"))
    test_ds = FlowerDataset(os.path.join(DATASET_DIR, "test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Classifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"🚀 Training on {device} ...")

    for epoch in range(args.epochs):
        loss, acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"📌 Epoch [{epoch+1}/{args.epochs}] | Loss: {loss:.4f} | Acc: {acc:.4f}")
        evaluate(model, val_loader, criterion, device, "Validation")

    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model.pth"))
    print("✅ Model saved!")

    print("🧪 Final Test:")
    evaluate(model, test_loader, criterion, device, "Test")


if __name__ == "__main__":
    main()
