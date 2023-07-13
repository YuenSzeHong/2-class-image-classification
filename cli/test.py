import os

# make a prediction for a new image.
from commons.test import load_image
from keras.models import load_model

# entry point, run the example
end = False
CLASS1_NAME = "cat"
CLASS2_NAME = "dog"
# load model

model = load_model('s207909_model.h5')

correct = 0
wrong = 0
count = 0

basedir = 'dogs_vs_cats/test/cats'
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
        print("Cats: Images {} Correct {} Wrong {} {}".format(count, correct, wrong, wrong_file))

basedir = 'dogs_vs_cats/test/dogs'

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
        print("Dogs: Images {} Correct {} Wrong {} {}".format(count, correct, wrong, wrong_file))
