import os

# make a prediction for a new image.
from commons.utils import load_image
from commons.config import CLASS1_NAME, CLASS2_NAME, DEFAULT_MODEL_FILENAME, TEST_DIR
from keras.models import load_model

# entry point, run the example

# load model
model = load_model(DEFAULT_MODEL_FILENAME)

correct = 0
wrong = 0
count = 0
basedir = os.path.join(TEST_DIR, CLASS1_NAME)
wrong_file = []
for file in os.listdir(basedir):
    count += 1
    img = load_image(os.path.join(basedir, file))
    # predict the class
    result = model.predict(img)
    # check result
    if result > 0.5:
        wrong_file.append(file)
        wrong += 1
    else:
        correct += 1
    if count % 10 == 0:
        print(f"{CLASS1_NAME.capitalize()}: Images {count} Correct {correct} Wrong {wrong} {wrong_file}")

basedir = os.path.join(TEST_DIR, CLASS2_NAME)

correct = 0
wrong = 0
count = 0
wrong_file = []
for file in os.listdir(basedir):
    count += 1
    img = load_image(os.path.join(basedir, file))
    # predict the class
    result = model.predict(img)
    # check result
    if result <= 0.5:
        wrong_file.append(file)
        wrong += 1
    else:
        correct += 1
    if count % 10 == 0:
        print(f"{CLASS2_NAME.capitalize()}: Images {count} Correct {correct} Wrong {wrong} {wrong_file}")
