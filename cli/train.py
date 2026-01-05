# -*- coding: utf-8 -*-
"""
Student ID:
Name:
"""

from commons.model import define_model
from keras.preprocessing.image import ImageDataGenerator
from commons.config import CLASS_NAMES, BATCH_SIZE, DEFAULT_EPOCHS, IMAGENET_MEAN, \
    DEFAULT_MODEL_FILENAME, IMAGE_SIZE, TRAIN_DIR

NUM_EPOCHS = DEFAULT_EPOCHS


# Training
def train():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = IMAGENET_MEAN
    # prepare training data
    train_data = datagen \
        .flow_from_directory(TRAIN_DIR, class_mode='binary',
                             batch_size=BATCH_SIZE, target_size=IMAGE_SIZE, classes=CLASS_NAMES)
    # fit model
    model.fit(train_data, steps_per_epoch=len(
        train_data), epochs=NUM_EPOCHS, verbose=1)
    # save model
    model.save(DEFAULT_MODEL_FILENAME)


# entry point, run the test harness
if __name__ == '__main__':
    train()
