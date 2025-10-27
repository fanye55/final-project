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
# Global Constants
# -----------------------------
DATA_URL = "https://dagshub.com/fanye55/final-project/raw/master/feature/dataset.tar.gz"
DATA_DIR = "dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = DEVICE  # ✅ pytest requires this


# -----------------------------
# Download dataset only when running locally
# -----------------------------
def download_dataset():
    if os.path.exists(DATA_DIR):
        print("✅ Dataset already exists")
        return

    print("📥 Downloading dataset...")
    tar_path = "dataset.tar.gz"
    urllib.request.urlretrieve(DATA_URL, tar_path)

    print("📦 Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()

    os.remove(tar_path)
    print("✅ Dataset ready")


# -----------------------------
# Dataset Loader
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
            feat = np.loadtxt(os.path.join(feature_dir, f))
            label = int(np.loadtxt(os.path.join(label_dir, l)))

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
# CNN Model (pytest requires)
# -----------------------------
class CNN_Model(nn.Module):
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
            nn.Linear(64, 3)  # ✅ 3 classes
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,4)→(B,1,4)
        return self.net(x)


# -----------------------------
# Init loaders ONLY if dataset exists
# -----------------------------
if os.path.exists(DATA_DIR):
    train_dataset = IrisTxtDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
else:
    train_dataset = None
    train_loader = None


# -----------------------------
# Evaluation function (pytest requires)
# -----------------------------
def evaluate_model(model, data_loader, desc="Eval"):
    if data_loader is None:
        print("⚠ No dataset available for evaluation")
        return 0

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            preds.extend(torch.argmax(model(X), dim=1).cpu().numpy())
            trues.extend(y.cpu().numpy())

    acc = accuracy_score(trues, preds)
    print(f"{desc} Accuracy: {acc:.4f}")
    return acc


# -----------------------------
# Main function (Only runs locally)
# -----------------------------
if __name__ == "__main__":
    download_dataset()

    train_dataset = IrisTxtDataset("train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = CNN_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training Started...")
    for epoch in range(3):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        evaluate_model(model, train_loader, f"Epoch {epoch+1}")

    print("✅ Training finished!")
