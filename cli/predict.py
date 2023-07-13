# make a prediction for a new image.
from keras.models import load_model

from commons.test import load_image

# entry point, run the example
end = False
CLASS1_NAME = "cat"
CLASS2_NAME = "dog"
# load model
model = load_model('s207909_model.h5')

while not end:

    try:
        file = input("enter a picture's file name (or exit to end):")
        if file != "exit":
            # load the image
            img = load_image(file)
            # predict the class
            result = model.predict(img)
            # check result
            if result > 0.5:
                print(result, f" {CLASS2_NAME.capitalize()}!")
            else:
                print(result, f" {CLASS1_NAME.capitalize()}!")
        else:
            end = True
    except OSError as e:
        print("Cannot Open File")
