"""T3-2 练习 3 起始代码：裸 print 的训练循环，Rich + structlog 改写目标。
"""
import random
import time

for epoch in range(50):
    time.sleep(0.1)  # 模拟一个 epoch 的训练
    loss = 2.5 / (epoch + 1) + random.random() * 0.05
    acc = 1 - loss / 3 + random.random() * 0.02
    lr = 0.001 * (0.95 ** epoch)
    print(f"epoch {epoch} loss={loss:.4f} acc={acc:.4f} lr={lr:.6f}")
print("训练完成")
