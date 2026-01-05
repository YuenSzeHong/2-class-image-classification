from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import SGD

from commons.config import INPUT_SHAPE, LEARNING_RATE, MOMENTUM


def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=INPUT_SHAPE)
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
