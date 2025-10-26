import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_dir = Path('./dataset')
result_dir = Path('./train-result')
result_dir.mkdir(exist_ok=True)

batch_size = 64
learning_rate = 0.001

global_scaler = StandardScaler()


class IrisDataset(Dataset):
    def __init__(self, feature_dir, label_dir, mode="train"):
        self.features = []
        self.labels = []

        feature_files = sorted(feature_dir.glob('*.txt'))
        label_files = sorted(label_dir.glob('*.txt'))

        for feature_file, label_file in zip(feature_files, label_files):
            feature = np.loadtxt(feature_file, delimiter="\t")
            label = int(np.loadtxt(label_file))
            self.features.append(feature)
            self.labels.append(label)

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        if mode == "train":
            self.features = global_scaler.fit_transform(self.features)
        else:
            self.features = global_scaler.transform(self.features)

        self.features = self.features.reshape(-1, 1, 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


train_dataset = IrisDataset(dataset_dir / 'train' / 'feature', dataset_dir / 'train' / 'label', "train")
val_dataset = IrisDataset(dataset_dir / 'val' / 'feature', dataset_dir / 'val' / 'label', "val")
test_dataset = IrisDataset(dataset_dir / 'test' / 'feature', dataset_dir / 'test' / 'label', "test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=2)
        self.fc1 = nn.Linear(16 * 2, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_model(epochs):
    model = CNN_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    result_file = result_dir / 'training_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Training Results for {epochs} epochs\n")
        f.write("---------------------------------\n")

    for epoch in range(epochs):
        model.train()
        running_loss, total, correct = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        train_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")

        with open(result_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: Loss {train_loss:.4f}, Acc {train_acc:.2f}%\n")

    return model


def evaluate_model(model,loader, title):
    model.eval()
    correct, total = 0, 0

    result_file = result_dir / 'training_results.txt'

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"{title} Accuracy: {acc:.2f}%")

    with open(result_file, 'a') as f:
        f.write(f"{title} Accuracy: {acc:.2f}%\n")

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    train_model(args.epochs)
    evaluate_model(val_loader, "Validation")
    evaluate_model(test_loader, "Test")


if __name__ == "__main__":
    main()
