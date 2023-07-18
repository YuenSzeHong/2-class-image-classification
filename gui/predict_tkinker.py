from tkinter import filedialog, font, Tk, Label, Button

from PIL import Image, ImageTk
from keras.models import load_model

from commons.test import load_image

CLASS1_NAME = "man"
CLASS2_NAME = "women"
IMAGE_DISPLAY_SIZE = 512


class MainWindow:
    def __init__(self, master):
        # set window title
        master.title("Prediction GUI")

        # create widgets
        self.image_label = Label(master, text="Image:")
        self.image_path_label = Label(master, text="")
        self.image_button = Button(master, text="Choose", command=self.choose_image)
        self.input_image_label = Label(master)
        self.model_label = Label(master, text="Model:")
        self.model_path_label = Label(master, text="")
        self.model_button = Button(master, text="Choose", command=self.choose_model)
        self.predict_button = Button(master, text="Predict", command=self.predict)
        self.result_label = Label(master, text="Result:")
        self.result_text_label = Label(master, text="")

        # create layout
        self.image_label.grid(row=0, column=0)
        self.image_path_label.grid(row=0, column=1)
        self.image_button.grid(row=0, column=2)
        self.input_image_label.grid(row=1, column=0, columnspan=3)
        self.model_label.grid(row=2, column=0)
        self.model_path_label.grid(row=2, column=1)
        self.model_button.grid(row=2, column=2)
        self.predict_button.grid(row=3, column=0, columnspan=3)
        self.result_label.grid(row=4, column=0)
        self.result_text_label.grid(row=5, column=2)

    def choose_image(self):
        # open file dialog to select image file
        file_path = filedialog.askopenfilename(
            title="Open Image", filetypes=[("Image Files", "*.png *.jpg *.bmp")]
        )
        if file_path:
            self.image_path_label.config(text=file_path)
            image = Image.open(file_path)

            # calculate aspect ratio of original image
            width, height = image.size
            aspect_ratio = width / height

            # resize image to fit within display area while preserving aspect ratio
            if width > height:
                new_width = IMAGE_DISPLAY_SIZE
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = IMAGE_DISPLAY_SIZE
                new_width = int(new_height * aspect_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.input_image_label.config(image=photo)
            self.input_image_label.image = photo

    def choose_model(self):
        # open file dialog to select model file
        file_path = filedialog.askopenfilename(
            title="Open Model", filetypes=[("H5 Files", "*.h5")]
        )
        if file_path:
            self.model_path_label.config(text=file_path)

    def predict(self):
        # get image path
        image_path = self.image_path_label.cget("text")

        # check if image path is valid
        if not image_path:
            self.result_text_label.config(text="Please choose an image.")
            return

        # load model
        model_path = self.model_path_label.cget("text")
        if not model_path:
            self.result_text_label.config(text="Please choose a model.")
            return
        model = load_model(model_path)

        # load image
        image = load_image(image_path)

        # predict class label
        result = model.predict(image)
        class_label = CLASS1_NAME if result < 0.5 else CLASS2_NAME

        # display predicted class label,
        # with the individual class probabilities
        self.result_text_label.config(
            text=f"{class_label.capitalize()} " +
                 f" ({CLASS1_NAME.capitalize()}: {(1 - result[0][0]):.2f}, " +
                 f"{CLASS2_NAME.capitalize()}: {result[0][0]:.2f})"
        )


if __name__ == "__main__":
    root = Tk()
    font.nametofont("TkDefaultFont").configure(
        size=12
    )
    window = MainWindow(root)
    root.mainloop()
