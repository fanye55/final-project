import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.app import CNN_Model, train_loader, train_model, evaluate_model, device

def test_dataloader():
    # 测试数据是否能正常取出
    inputs, labels = next(iter(train_loader))
    assert inputs.shape[-1] == 4  # 特征维度是否正确
    assert len(labels.shape) == 1  # 标签是否是一维
    assert inputs.size(0) == labels.size(0)

def test_forward_pass():
    # 测试模型能否前向传播
    model = CNN_Model().to(device)
    inputs, _ = next(iter(train_loader))
    inputs = inputs.to(device)
    outputs = model(inputs)
    assert outputs.shape[-1] == 3  # 输出类别数量是否正确

def test_one_epoch_train():
    # 只跑一轮训练，确保不会报错
    model = CNN_Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0  # loss 应为正数

def test_evaluate():
    # 只测试函数能正常运行
    val_acc = evaluate_model(train_loader, "Test_in_pytest")
    assert 0 <= val_acc <= 100  # 准确率必须合理范围
