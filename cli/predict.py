from keras.models import load_model
from commons.config import CLASS1_NAME, CLASS2_NAME, DEFAULT_MODEL_FILENAME
from commons.utils import load_image

# entry point, run the example
end = False

# load model
model = load_model(DEFAULT_MODEL_FILENAME)

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
    except OSError:
        print("Cannot Open File")
