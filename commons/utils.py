from keras.src.utils import load_img, img_to_array
from commons.config import IMAGENET_MEAN, IMAGE_SIZE


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=IMAGE_SIZE)
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    # center pixel data
    img = img.astype('float32')
    img = img - IMAGENET_MEAN
    return img
