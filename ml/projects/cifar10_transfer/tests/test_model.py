"""model.py 的单元测试：LeNet 前向形状、ResNet18 冻结策略、优化器分组学习率。

运行：在本项目根目录执行 `../../../.venv/bin/python -m pytest tests/ -v`

说明：resnet18 相关测试一律用 pretrained=False（随机初始化），
不触发 45MB 权重下载——结构/冻结行为与是否预训练无关。
"""

import pytest

torch = pytest.importorskip("torch", reason="本项目需要 torch/torchvision")

from model import FINETUNE_MODES, MODEL_NAMES, LeNet, count_trainable_params, create_model, create_optimizer

BASE_CONFIG = {
    "model": {"name": "lenet", "finetune": "head_only", "pretrained": False, "num_classes": 10},
    "train": {"lr_head": 1e-3, "lr_backbone_deep": 1e-4, "lr_backbone_shallow": 1e-5},
}


def _config(**model_overrides):
    config = {"model": {**BASE_CONFIG["model"], **model_overrides},
              "train": dict(BASE_CONFIG["train"])}
    return config


# ---------------------------------------------------------------- LeNet（参考实现）


def test_lenet_前向形状():
    model = LeNet(num_classes=10)
    x = torch.randn(2, 3, 32, 32)          # NCHW：2 张 3 通道 32×32
    out = model(x)
    assert out.shape == (2, 10)            # 输出 logits（不 softmax！前置文档 §5.3）


def test_lenet_工厂创建():
    model = create_model(_config(name="lenet"))
    assert isinstance(model, LeNet)


# ---------------------------------------------------------------- ResNet18 迁移学习手术


def test_resnet18_换头():
    model = create_model(_config(name="resnet18"))
    assert model.fc.out_features == 10     # 头换成 10 类（原来是 ImageNet 的 1000 类）
    assert model.fc.in_features == 512


def test_resnet18_head_only_冻结():
    """冻结主干、只训新头：可训练参数应该恰好是 fc 头的 512×10+10 = 5130。"""
    model = create_model(_config(name="resnet18", finetune="head_only"))
    assert all(p.requires_grad for p in model.fc.parameters())
    assert not model.conv1.weight.requires_grad, "主干必须冻结（前置文档 §6.3）"
    assert not model.layer4[0].conv1.weight.requires_grad
    assert count_trainable_params(model) == 512 * 10 + 10


def test_resnet18_full_全部放开():
    model = create_model(_config(name="resnet18", finetune="full"))
    assert all(p.requires_grad for p in model.parameters())
    assert count_trainable_params(model) > 11_000_000


def test_resnet18_未知finetune报错():
    with pytest.raises(ValueError):
        create_model(_config(name="resnet18", finetune="magic"))


def test_未知模型名报错():
    with pytest.raises(ValueError, match="未知模型"):
        create_model(_config(name="vgg999"))


# ---------------------------------------------------------------- create_optimizer


def test_优化器_只收活参数_head_only():
    """冻结参数不进优化器（前置文档 §6.3 第④步）。"""
    config = _config(name="resnet18", finetune="head_only")
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    n_params = sum(len(g["params"]) for g in optimizer.param_groups)
    n_alive = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_params == n_alive


def test_优化器_full微调_分组学习率():
    """头 1e-3、layer4 1e-4、浅层默认 1e-5（前置文档 §6.4）。"""
    config = _config(name="resnet18", finetune="full")
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    lrs = sorted(g["lr"] for g in optimizer.param_groups)
    assert lrs == [1e-5, 1e-4, 1e-3]     # 浅层(默认) / 深层 / 新头 各一组


def test_优化器_lenet_单一学习率():
    model = create_model(_config(name="lenet"))
    optimizer = create_optimizer(model, _config(name="lenet"))
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_模型名单_与文档一致():
    assert set(MODEL_NAMES) == {"lenet", "resnet18"}
    assert set(FINETUNE_MODES) == {"head_only", "full"}
