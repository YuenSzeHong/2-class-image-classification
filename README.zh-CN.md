# 二分图片分类

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

## 设置

### 创建虚拟环境

```bash
python3 -m venv .venv
```

如果你是在 Windows 上，把 `python3` 改成 `python` 或 `py`。

### 激活虚拟环境

```bash
source .venv/bin/activate
```

如果你是在 Windows 上

```pwsh
.venv\Scripts\activate
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载数据集

你可以使用任何你想要的数据集，只要文件结构正确就行

```text
data_root
├── train
│   ├── class_1
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   └── class_2
│       ├── image_1.jpg
│       ├── image_2.jpg
│       └── ...
└── test
    ├── class_1
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    └── class_2
        ├── image_1.jpg
        ├── image_2.jpg
        └── ...
```

其中 `class_1` 和 `class_2` 是你想要分类的类别的名字，并且在所有脚本中更改 `CLASS1_NAME` 和 `CLASS2_NAME`。图片文件名不重要。只要它们是 `.jpg` 或 `.png` 文件，它们就会被加载。

### 训练

在 `gui` 文件夹中运行 `train_gui.py`，并按照说明操作。

### 测试

在 `gui` 文件夹中运行 `test_gui.py`，并按照说明操作。

### 评估测试结果

测试结果显示在 GUI 上，查看分数。如果你对分数不满意，可以尝试用更多的 epochs 和更多的数据重新训练模型。

#### 更改 epochs 数量

你可以通过更改 `train_gui.py` 中的 `NUM_EPOCHS` 变量来更改 epochs 数量。**不建议**将其设置为非常高的数字，因为它将花费很长时间来训练，而且分数不会有太大的提高。

### 预测

在 `gui` 文件夹中运行 `predict_gui.py`，并按照说明操作。

### 预测结果不正确？

如果预测结果不正确，你可以尝试用更多的 epochs 和更多的数据重新训练模型。或者由于系统限制，模型是用缩小的图像训练的，所以预测结果可能不准确。
