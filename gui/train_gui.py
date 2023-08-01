from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, \
    QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout, QProgressBar
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.src.callbacks import Callback

from commons.model import define_model

NUM_EPOCHS = 3
CLASS1_NAME = "cat"
CLASS2_NAME = "dog"


class UpdateWidgetsCallback(Callback, QObject):
    update_signal = pyqtSignal(str)

    def __init__(self, progress_bar, result_text):
        super(Callback, self).__init__()
        super(QObject, self).__init__()
        self.progress_bar = progress_bar
        self.result_text = result_text

    def on_train_begin(self, logs=None):
        if logs is not None:
            self.progress_bar.setMaximum(
                self.params["steps"] * self.params["epochs"])
            self.progress_bar.setValue(0)
            self.update_signal.emit('Training started...')

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            self.progress_bar.setValue(batch + 1)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            message = f'Epoch {epoch + 1}/{self.params["epochs"]}: loss={logs["loss"]:.4f}, acc={logs["accuracy"]:.4f}'
            self.progress_bar.setValue(
                self.progress_bar.value() + self.params["steps"] * (epoch + 1))
            self.update_signal.emit(message)


class TrainThread(QThread):
    finished = pyqtSignal(float, float)

    def __init__(self, model_path, data_dir, update_widgets):
        super().__init__()
        self.model_path = model_path
        self.data_dir = data_dir
        self.update_widgets = update_widgets

    @pyqtSlot()
    def run(self):
        # define model
        model = define_model()
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]
        # prepare iterator
        train_it = datagen \
            .flow_from_directory(self.data_dir, class_mode='binary',
                                 batch_size=64,
                                 target_size=(224, 224),
                                 classes=[CLASS1_NAME, CLASS2_NAME])

        # fit model
        training = model.fit(train_it, steps_per_epoch=len(train_it),
                             epochs=NUM_EPOCHS, verbose=0,
                             callbacks=[self.update_widgets])

        # save model
        model.save(self.model_path)
        # get final accuracy and loss
        final_acc = training.history['accuracy'][-1]
        final_loss = training.history['loss'][-1]
        self.finished.emit(final_acc, final_loss)


class MainWindow(QWidget):

    @pyqtSlot(int, int, int, list)
    def __init__(self):
        super().__init__()

        self.train_thread = None
        self.setWindowTitle(
            f"Train {CLASS1_NAME.capitalize()} and {CLASS2_NAME.capitalize()}")

        # create widgets
        self.title_label = QLabel(
            f"{CLASS1_NAME.capitalize()} vs " +
            f"{CLASS2_NAME.capitalize()} Trainer")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.data_label = QLabel("Data Directory:")
        self.data_path_label = QLabel("")
        self.data_button = QPushButton("Choose")

        self.model_label = QLabel("Model Path:")
        self.model_path_label = QLabel("")
        self.model_button = QPushButton("Choose")

        self.train_button = QPushButton("Train Model")
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar()

        self.result_label = QLabel("Training Result:")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.update_widgets = UpdateWidgetsCallback(
            self.progress_bar, self.result_text)

        # create layouts
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_label)
        data_layout.addWidget(self.data_path_label)
        data_layout.addWidget(self.data_button)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.model_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(data_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_text)

        # connect buttons to functions
        self.data_button.clicked.connect(self.choose_data)
        self.model_button.clicked.connect(self.choose_model)
        self.train_button.clicked.connect(self.train_model)
        self.stop_button.clicked.connect(self.stop_training)
        self.update_widgets.update_signal.connect(self.result_text.append)

    def choose_data(self):
        # open file dialog to select data directory
        dir_path = QFileDialog.getExistingDirectory(
            self, "Open Data Directory")
        if dir_path:
            self.data_path_label.setText(dir_path)

    def choose_model(self):
        # open file dialog to select model path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "H5 Files (*.h5)")
        if file_path:
            self.model_path_label.setText(file_path)

    @pyqtSlot()
    def train_model(self):
        # get data directory and model path
        data_dir = self.data_path_label.text()
        model_path = self.model_path_label.text()
        if not data_dir:
            self.result_text.append("Please choose a data directory.")
            return
        if not model_path:
            self.result_text.append("Please choose a model path.")
            return
        # start train thread
        self.train_thread = TrainThread(
            model_path, data_dir, self.update_widgets)
        self.train_thread.finished.connect(self.train_finished)
        self.train_thread.start()
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_training(self):
        self.train_thread.terminate()
        self.result_text.append("Training stopped.")
        # enable train button and disable stop button
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def train_finished(self, accuracy, loss):
        self.result_text.append(f"Training finished. " +
                                f"Accuracy: {accuracy}, Loss: {loss}")
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
