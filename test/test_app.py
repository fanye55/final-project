import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.app import CNN_Model, train_loader, evaluate_model, device

def test_dataloader():
    inputs, labels = next(iter(train_loader))
    assert inputs.shape[-1] == 4
    assert len(labels.shape) == 1
    assert inputs.size(0) == labels.size(0)

def test_forward_pass():
    model = CNN_Model().to(device)
    inputs, _ = next(iter(train_loader))
    outputs = model(inputs.to(device))
    assert outputs.shape[-1] == 3

def test_one_epoch_train():
    model = CNN_Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    inputs, labels = next(iter(train_loader))
    optimizer.zero_grad()
    outputs = model(inputs.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()

    assert loss.item() > 0

def test_evaluate():
    model = CNN_Model().to(device)
    acc = evaluate_model(model, train_loader, "Test_in_pytest")
    assert 0 <= acc <= 100
