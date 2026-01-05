# 二分圖片分類

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

## 設置

### 創建虛擬環境

```bash
python3 -m venv venv
```

如果你是在 Windows 上，把 `python3` 改成 `python` 或 `py`。

### 激活虛擬環境

```bash
source venv/bin/activate
```

如果你是在 Windows 上

```pwsh
venv\Scripts\activate
```

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 下載數據集

你可以使用任何你想要的數據集，只要文件結構正確就行

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

其中 `class_1` 和 `class_2` 是你想要分類的類別的名字。你可以在 `commons/config.py` 中更改 `CLASS1_NAME` 和 `CLASS2_NAME`。圖片文件名不重要。只要它們是 `.jpg` 或 `.png` 文件，它們就會被加載。

### 訓練

在 `gui` 文件夾中運行 `train_gui.py`，並按照說明操作。

### 測試

在 `gui` 文件夾中運行 `test_gui.py`，並按照說明操作。

### 評估測試結果

測試結果顯示在 GUI 上，查看分數。如果你對分數不滿意，可以嘗試用更多的 epochs 和更多的數據重新訓練模型。

#### 更改 epochs 數量

你可以通過更改 `commons/config.py` 中的 `DEFAULT_EPOCHS` 變量來更改 epochs 數量。**不建議**將其設置為非常高的數字，因為它將花費很長時間來訓練，而且分數不會有太大的提高。

### 預測

在 `gui` 文件夾中運行 `predict_gui.py`，並按照說明操作。

### 預測結果不正確？

如果預測結果不正確，你可以嘗試用更多的 epochs 和更多的數據重新訓練模型。或者由於系統限制，模型是用縮小的圖片訓練的，所以預測結果可能不準確。
