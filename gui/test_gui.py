import os

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import \
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QTextEdit, \
    QVBoxLayout, QHBoxLayout, QProgressBar
from tensorflow.keras.models import load_model
from commons.config import CLASS1_NAME, CLASS2_NAME
from commons.utils import load_image


class TestThread(QThread):
    started = pyqtSignal(int)
    finished = pyqtSignal(int, int, int, list)
    progress = pyqtSignal(int)

    def __init__(self, model_path, test_dir):
        super().__init__()
        self.model_path = model_path
        self.test_dir = test_dir

    @pyqtSlot()
    def run(self):
        # load model
        model = load_model(self.model_path)
        # prepare test data
        correct = 0
        wrong = 0
        count = 0
        wrong_files = []
        total_files = len(os.listdir(os.path.join(self.test_dir, CLASS1_NAME))) + len(
            os.listdir(os.path.join(self.test_dir, CLASS2_NAME)))
        self.started.emit(total_files)
        for file in os.listdir(os.path.join(self.test_dir, CLASS1_NAME)):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(self.test_dir, CLASS1_NAME, file)
                img = load_image(img_path)
                result = model.predict(img)
                if result[0] < 0.5:
                    correct += 1
                else:
                    wrong += 1
                    wrong_files.append(file)
                count += 1
                self.progress.emit(count)

        for file in os.listdir(os.path.join(self.test_dir, CLASS2_NAME)):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(self.test_dir, CLASS2_NAME, file)
                img = load_image(img_path)
                result = model.predict(img)
                if result[0] >= 0.5:
                    correct += 1
                else:
                    wrong += 1
                    wrong_files.append(file)
                count += 1
                self.progress.emit(count)
        # emit signal with test result
        self.finished.emit(correct, wrong, count, wrong_files)


class MainWindow(QWidget):

    @pyqtSlot()
    def __init__(self):
        super().__init__()

        self.test_thread = None
        self.setWindowTitle(f"{CLASS1_NAME} vs {CLASS2_NAME} Tester")

        # create widgets
        self.title_label = QLabel(
            f"{CLASS1_NAME} vs {CLASS2_NAME} Tester", self)
        self.title_label.setAlignment(Qt.AlignCenter)

        self.model_label = QLabel("Model Path:", self)
        self.model_path_label = QLabel("", self)
        self.model_button = QPushButton("Choose", self)

        self.test_label = QLabel("Test Directory:", self)
        self.test_path_label = QLabel("", self)
        self.test_button = QPushButton("Choose", self)

        self.run_button = QPushButton("Run Test", self)

        self.result_label = QLabel("Test Result:", self)
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        # create layouts
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.model_button)

        test_layout = QHBoxLayout()
        test_layout.addWidget(self.test_label)
        test_layout.addWidget(self.test_path_label)
        test_layout.addWidget(self.test_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(test_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.result_text)

        # connect buttons to functions
        self.model_button.clicked.connect(self.choose_model)
        self.test_button.clicked.connect(self.choose_test)
        self.run_button.clicked.connect(self.run_test)

    def choose_model(self):
        # open file dialog to select model file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "", "H5 Files (*.h5)")
        if file_path:
            self.model_path_label.setText(file_path)

    def choose_test(self):
        # open file dialog to select test directory
        dir_path = QFileDialog.getExistingDirectory(
            self, "Open Test Directory", "")
        if dir_path:
            self.test_path_label.setText(dir_path)

    def run_test(self):
        # get model path and test directory
        model_path = self.model_path_label.text()
        test_dir = self.test_path_label.text()
        if not model_path:
            self.result_text.append("Please choose a model path.")
            return
        if not test_dir:
            self.result_text.append("Please choose a test directory.")
            return
        # start test thread
        self.test_thread = TestThread(model_path, test_dir)
        self.test_thread.finished.connect(self.test_finished)
        self.test_thread.started.connect(self.progress_bar.setMaximum)
        self.test_thread.progress.connect(self.progress_bar.setValue)
        self.test_thread.start()

    def test_finished(self, correct, wrong, count, wrong_files):
        self.result_text.append(f"Correct: {correct}")
        self.result_text.append(f"Wrong: {wrong}")
        self.result_text.append(f"Total: {count}")
        self.result_text.append(f"Accuracy: {correct / count * 100}%")
        self.result_text.append(f"Wrong Files: {wrong_files}")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
