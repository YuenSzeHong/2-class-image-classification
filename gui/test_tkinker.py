import os
from tkinter import filedialog, messagebox, scrolledtext, font, Tk, Label, Button, END
from tensorflow.keras.models import load_model
from commons.test import load_image

CLASS1_NAME = "cat"
CLASS2_NAME = "dog"


class TestThread:
    def __init__(self, model_path, test_dir):
        self.model_path = model_path
        self.test_dir = test_dir

    def run(self):
        # load model
        model = load_model(self.model_path)
        # prepare test data
        correct = 0
        wrong = 0
        count = 0
        wrong_files = []
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
        # display test result
        messagebox.showinfo(
            "Test Result",
            f"Correct: {correct}\nWrong: {wrong}\nTotal: {count}\nAccuracy: {correct / count * 100}%\nWrong Files: {wrong_files}"
        )


class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title(f"{CLASS1_NAME} vs {CLASS2_NAME} Tester")

        # create widgets
        self.title_label = Label(
            master, text=f"{CLASS1_NAME} vs {CLASS2_NAME} Tester")
        self.title_label.pack()

        self.model_label = Label(master, text="Model Path:")
        self.model_label.pack()
        self.model_path_label = Label(master, text="")
        self.model_path_label.pack()
        self.model_button = Button(
            master, text="Choose", command=self.choose_model)
        self.model_button.pack()

        self.test_label = Label(master, text="Test Directory:")
        self.test_label.pack()
        self.test_path_label = Label(master, text="")
        self.test_path_label.pack()
        self.test_button = Button(
            master, text="Choose", command=self.choose_test)
        self.test_button.pack()

        self.run_button = Button(
            master, text="Run Test", command=self.run_test)
        self.run_button.pack()

        self.result_label = Label(master, text="Test Result:")
        self.result_label.pack()
        self.result_text = scrolledtext.ScrolledText(master, width=50, height=10)
        self.result_text.pack()

    def choose_model(self):
        # open file dialog to select model file
        file_path = filedialog.askopenfilename(
            title="Open Model", filetypes=[("H5 Files", "*.h5")])
        if file_path:
            self.model_path_label.config(text=file_path)

    def choose_test(self):
        # open file dialog to select test directory
        dir_path = filedialog.askdirectory(title="Open Test Directory")
        if dir_path:
            self.test_path_label.config(text=dir_path)

    def run_test(self):
        # get model path and test directory
        model_path = self.model_path_label.cget("text")
        test_dir = self.test_path_label.cget("text")
        if not model_path:
            self.result_text.insert(END, "Please choose a model path.\n")
            return
        if not test_dir:
            self.result_text.insert(END, "Please choose a test directory.\n")
            return
        # start test thread
        test_thread = TestThread(model_path, test_dir)
        test_thread.run()


if __name__ == "__main__":
    root = Tk()
    font.nametofont("TkDefaultFont").configure(
        size=12
    )
    window = MainWindow(root)
    root.mainloop()
