
# Configuration for 2-class-image-classification

CLASS_NAMES = ["cat", "dog"]
CLASS1_NAME = CLASS_NAMES[0]
CLASS2_NAME = CLASS_NAMES[1]

# Image settings
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# ImageNet mean values for centering
IMAGENET_MEAN = [123.68, 116.779, 103.939]

# Training settings
BATCH_SIZE = 64
DEFAULT_EPOCHS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# Model settings
DEFAULT_MODEL_FILENAME = "model.h5"

# Data settings
# Assuming 'data' folder in project root if running from root
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
