# Two Category Image Classification

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

## Setup

### Create Virtual Environment

```bash
python3 -m venv venv
```

Change `python3` to `python` or `py` if you are on Windows.

### Activate Virtual Environment

```bash
source venv/bin/activate
```

If you are on Windows

```pwsh
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download dataset

You can pretty much use any dataset you want, as long as the file structure is correct

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

Where `class_1` and `class_2` are the names of the classes you want to classify, and change `CLASS1_NAME` and `CLASS2_NAME` in all scripts accordingly. Image filenames do not matter. As long as they are `.jpg` or `.png` files, they will be loaded.

### Train

run `train_gui.py` in the `gui` folder and follow the instructions.

### Test

run `test_gui.py` in the `gui` folder and follow the instructions.

### Evaluate Test Results

the test results are displayed on the GUI, look at the score. If you are not satisfied with the score, you can try to train the model again with more epochs and probably more data.

#### Change number of epochs

You can change the number of epochs by changing the `NUM_EPOCHS` variable in `train_gui.py`. It is **NOT** recommended to set it to a very high number, because it will take a long time to train and the score will not improve much.

### Predict

run `predict_gui.py` in the `gui` folder and follow the instructions.

### The Predict result is not correct?

If the predict result is not correct, you can try to train the model again with more epochs and probably more data. Or due to system limitations, the model is trained with scaled down images, so the predict result may not be accurate.
