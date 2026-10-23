"""T2-1 练习 2 起始代码：把硬编码配置换成 pydantic-settings。

现状：配置以模块级常量存在，改配置 = 改代码 + 改 git 历史。
练习目标：换成 TrainSettings(BaseSettings)，环境变量 + .env 驱动，
配置错误在启动瞬间暴露。
"""

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = "cuda"        # 试试 "cuda " 或 "CUDA" 原版会怎样——静默失效
OUTPUT_DIR = "runs/exp01"
USE_AMP = True


def train() -> None:
    print(f"训练开始: lr={LEARNING_RATE}, batch={BATCH_SIZE}, device={DEVICE}")
    for epoch in range(EPOCHS):
        fake_loss = 2.5 / (epoch + 1)
        print(f"epoch {epoch}: loss={fake_loss:.4f}")
    print(f"产物将写入 {OUTPUT_DIR}, use_amp={USE_AMP}")


if __name__ == "__main__":
    train()
