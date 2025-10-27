import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.app import CNN_Model, device, evaluate_model

# ✅ 使用随机数据代替真实 dataset
def get_fake_data(batch_size=32):
    # 随机生成特征数据：形状 (batch, 4)
    inputs = torch.randn(batch_size, 4)
    # 随机生成标签数据：范围 [0,2] → 假设3分类
    labels = torch.randint(0, 3, (batch_size,))
    return inputs, labels


def test_dataloader():
    inputs, labels = get_fake_data()
    assert inputs.shape == (32, 4)
    assert labels.shape == (32,)


def test_forward_pass():
    model = CNN_Model().to(device)
    inputs, _ = get_fake_data()
    outputs = model(inputs.to(device))
    assert outputs.shape[-1] == 3  # 3类分类任务


def test_one_epoch_train():
    model = CNN_Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    inputs, labels = get_fake_data()
    optimizer.zero_grad()
    outputs = model(inputs.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


def test_evaluate():
    model = CNN_Model().to(device)

    # ✅ 模拟 dataloader：返回随机数据
    fake_loader = [(get_fake_data()) for _ in range(3)]

    acc = evaluate_model(model, fake_loader, "Test_in_pytest")
    assert 0 <= acc <= 100
