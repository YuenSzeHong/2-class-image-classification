from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import \
    QApplication, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
from keras.models import load_model

from commons.test import load_image

CLASS1_NAME = "man"
CLASS2_NAME = "women"
IMAGE_DISPLAY_SIZE = 512


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # set window title
        self.setWindowTitle("Prediction GUI")

        # create widgets
        self.image_label = QLabel("Image:")
        self.image_path_label = QLabel("")
        self.image_button = QPushButton("Choose")
        self.input_image_label = QLabel()
        self.model_label = QLabel("Model:")
        self.model_path_label = QLabel("")
        self.model_button = QPushButton("Choose")
        self.predict_button = QPushButton("Predict")
        self.result_label = QLabel("Result:")
        self.result_text_label = QLabel("")

        # create layout
        layout = QGridLayout(self)
        layout.addWidget(self.image_label, 0, 0)
        layout.addWidget(self.image_path_label, 0, 1)
        layout.addWidget(self.image_button, 0, 2)
        # center align the image, and make it take up full width of window
        layout.addWidget(self.input_image_label, 1, 0, 1, 3, Qt.AlignCenter)
        layout.addWidget(self.model_label, 2, 0)
        layout.addWidget(self.model_path_label, 2, 1)
        layout.addWidget(self.model_button, 2, 2)
        layout.addWidget(self.predict_button, 3, 0, 1, 3)
        layout.addWidget(self.result_label, 4, 0)
        layout.addWidget(self.result_text_label, 5, 2)

        # connect buttons to functions
        self.image_button.clicked.connect(self.choose_image)
        self.model_button.clicked.connect(self.choose_model)
        self.predict_button.clicked.connect(self.predict)

    def choose_image(self):
        # open file dialog to select image file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image_path_label.setText(file_path)
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(
                IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE, Qt.KeepAspectRatio)
            self.input_image_label.setPixmap(pixmap)

    def choose_model(self):
        # open file dialog to select model file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "", "H5 Files (*.h5)")
        if file_path:
            self.model_path_label.setText(file_path)

    def predict(self):
        # get image path
        image_path = self.image_path_label.text()

        # check if image path is valid
        if not image_path:
            self.result_text_label.setText("Please choose an image.")
            return

        # load model
        model_path = self.model_path_label.text()
        if not model_path:
            self.result_text_label.setText("Please choose a model.")
            return
        model = load_model(model_path)

        # load image
        image = load_image(image_path)

        # predict class label
        result = model.predict(image)
        class_label = CLASS1_NAME if result < 0.5 else CLASS2_NAME

        # display predicted class label,
        # with the individual class probabilities
        self.result_text_label.setText(
            f"{class_label.capitalize()} " +
            f" ({CLASS1_NAME.capitalize()}: {(1 - result[0][0]):.2f}, " +
            f"{CLASS2_NAME.capitalize()}: {result[0][0]:.2f})")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
