import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk
from keras.models import load_model

from commons.test import load_image

CLASS1_NAME = "cat"
CLASS2_NAME = "dog"
IMAGE_DISPLAY_SIZE = 512


class MainWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)

        # set window title
        self.master.title("Prediction GUI")

        # create widgets
        self.image_label = tk.Label(self, text="Image:")
        self.image_path_label = tk.Label(self, text="")
        self.image_button = tk.Button(self, text="Choose", command=self.choose_image)
        self.input_image_label = tk.Label(self)
        self.model_label = tk.Label(self, text="Model:")
        self.model_path_label = tk.Label(self, text="")
        self.model_button = tk.Button(self, text="Choose", command=self.choose_model)
        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.result_label = tk.Label(self, text="Result:")
        self.result_text_label = tk.Label(self, text="")

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
            title="Open Image", filetypes=[("Image Files", "*.png;*.jpg;*.bmp")]
        )
        if file_path:
            self.image_path_label.config(text=file_path)
            image = Image.open(file_path)
            image = image.resize((IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE), Image.ANTIALIAS)
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
                 f" ({CLASS1_NAME.capitalize()}: {result}, " +
                 f"{CLASS2_NAME.capitalize()}: {1 - result})"
        )


if __name__ == "__main__":
    root = tk.Tk()
    font.nametofont("TkDefaultFont").configure(
        size=12
    )
    window = MainWindow(root)
    window.pack()
    root.mainloop()
