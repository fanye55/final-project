import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.datasets import load_iris

# Config
n_samples = 3000
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
random_seed = 123
np.random.seed(random_seed)

# Load Iris dataset
iris = load_iris()
X = iris['data']
y = iris['target']
classes = np.unique(y)
feature_names = iris['feature_names']

# Class-wise mean and covariance
class_stats = {}
for cls in classes:
    Xc = X[y == cls]
    mu = Xc.mean(axis=0)
    cov = np.cov(Xc, rowvar=False) * 1.15
    class_stats[cls] = (mu, cov)

# Generate dataset
samples, labels = [], []
for _ in range(n_samples):
    cls = np.random.choice(classes)
    mu, cov = class_stats[cls]
    try:
        s = np.random.multivariate_normal(mu, cov)
    except:
        s = np.random.normal(mu, np.sqrt(np.diag(cov)))
    samples.append(s)
    labels.append(cls)

df = pd.DataFrame(samples, columns=feature_names)
df["label"] = labels
df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Split dataset
train_end = int(n_samples * train_ratio)
val_end = int(n_samples * (train_ratio + val_ratio))

splits = {
    "train": df.iloc[:train_end],
    "val": df.iloc[train_end:val_end],
    "test": df.iloc[val_end:]
}

# Create directory structure
base_dir = Path("./dataset")
base_dir.mkdir(exist_ok=True)

for split, data in splits.items():
    split_dir = base_dir / split
    feature_dir = split_dir / "feature"
    label_dir = split_dir / "label"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in data.iterrows():
        file_id = f"{idx + 1:04d}"

        feature_file = feature_dir / f"feature_{file_id}.txt"
        label_file = label_dir / f"label_{file_id}.txt"

        # Save features (4 numbers in one row, tab separated)
        np.savetxt(feature_file, row[feature_names].values.reshape(1, -1),
                   fmt="%.4f", delimiter="\t")

        # Save label
        with open(label_file, "w") as f:
            f.write(str(int(row['label'])))

print("✅ 数据成功生成！")
print("📂 目录位于：", base_dir.resolve())
