# -*- coding: utf-8 -*-
"""
Student ID:
Name:
"""

from commons.model import define_model
from keras.preprocessing.image import ImageDataGenerator

NUM_EPOCHS = 1
CLASS1_NAME = "cat"
CLASS2_NAME = "dog"


# Training
def train():
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare training data
    train_data = datagen \
        .flow_from_directory('dogs_vs_cats/train', class_mode='binary',
                             batch_size=64, target_size=(224, 224), classes=[CLASS1_NAME, CLASS2_NAME])
    # fit model
    model.fit_generator(train_data, steps_per_epoch=len(
        train_data), epochs=1, verbose=1)
    # save model
    model.save('s207909_model.h5')


# entry point, run the test harness
if __name__ == '__main__':
    train()
