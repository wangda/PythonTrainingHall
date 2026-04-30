import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 造假数据
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))

# 2. 数据加载器
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 模型
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 4. 训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 5. 推理
model.eval()
with torch.no_grad():
    test_x = torch.randn(5, 20)
    pred = model(test_x)
    print(torch.softmax(pred, dim=1))
