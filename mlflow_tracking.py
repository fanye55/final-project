import os
import sys
import time
import argparse
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根到 sys.path，确保能 import app.app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.app import download_dataset, IrisTxtDataset, DATA_DIR, device, CNN_Model

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    raise ImportError("Please install mlflow: pip install mlflow mlflow-pytorch")

# 配置你的 DagsHub MLflow URI
DEFAULT_MLFLOW_URI = "https://dagshub.com/fanye55/final-project.mlflow"

def setup_mlflow(experiment_name: str, tracking_uri: str | None, token: str | None):
    if token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
    uri = tracking_uri or DEFAULT_MLFLOW_URI
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found at {uri}. Please create it in DagsHub.")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow URI: {uri}")
    print(f"Experiment: {experiment_name}")

def get_dataloader(batch_size: int = 16):
    if not os.path.exists(DATA_DIR):
        download_dataset()
        time.sleep(0.5)
    dataset = IrisTxtDataset("train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_and_log(epochs=3, batch_size=16, lr=0.001,
                  experiment="iris-cnn", tracking_uri=None, token=None):
    setup_mlflow(experiment, tracking_uri, token)
    loader = get_dataloader(batch_size)
    model = CNN_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=f"{experiment}_{int(time.time())}"):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)

        for epoch in range(1, epochs + 1):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0

            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item()

            acc = correct / total
            avg_loss = loss_sum / len(loader)

            print(f"Epoch {epoch}/{epochs} — loss={avg_loss:.4f} acc={acc:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)

        # 保存模型为 artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "model.pth"
            torch.save(model.state_dict(), fp)
            mlflow.log_artifact(str(fp), artifact_path="model")

    print("Training complete. Logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--experiment", type=str, default="iris-cnn")
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    train_and_log(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        experiment=args.experiment,
        tracking_uri=args.tracking_uri,
        token=args.token
    )
