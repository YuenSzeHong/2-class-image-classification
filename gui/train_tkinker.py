import tkinter as tk
from tkinter import filedialog, messagebox, font
from keras.preprocessing.image import ImageDataGenerator
from commons.model import define_model

NUM_EPOCHS = 1
CLASS1_NAME = "cat"
CLASS2_NAME = "dog"


class TrainThread:
    def __init__(self, model_path, data_dir):
        self.model_path = model_path
        self.data_dir = data_dir

    def run(self):
        model = define_model()
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=True)
        # specify imagenet mean values for centering
        datagen.mean = [123.68, 116.779, 103.939]
        # prepare iterator
        train_it = datagen \
            .flow_from_directory(self.data_dir, class_mode='binary',
                                 batch_size=64,
                                 classes=[CLASS1_NAME, CLASS2_NAME])
        # fit model
        model.fit(train_it, epochs=NUM_EPOCHS, verbose=0)
        # save model
        model.save(self.model_path)
        messagebox.showinfo("Training Result", "Training finished.")


class MainWindow:
    def __init__(self, master):
        self.master = master
        self.train_thread = None
        master.title(f"Train {CLASS1_NAME.capitalize()} and {CLASS2_NAME.capitalize()}")

        # create widgets
        self.title_label = tk.Label(
            master, text=f"{CLASS1_NAME.capitalize()} vs " +
                         f"{CLASS2_NAME.capitalize()} Trainer")
        self.title_label.pack()

        self.data_label = tk.Label(master, text="Data Directory:")
        self.data_label.pack()
        self.data_path_label = tk.Label(master, text="")
        self.data_path_label.pack()
        self.data_button = \
            tk.Button(master, text="Choose", command=self.choose_data)
        self.data_button.pack()

        self.model_label = tk.Label(master, text="Model Path:")
        self.model_label.pack()
        self.model_path_label = tk.Label(master, text="")
        self.model_path_label.pack()
        self.model_button = \
            tk.Button(master, text="Choose", command=self.choose_model)
        self.model_button.pack()

        self.train_button = \
            tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack()

    def choose_data(self):
        # open file dialog to select data directory
        dir_path = filedialog.askdirectory(title="Open Data Directory")
        if dir_path:
            self.data_path_label.config(text=dir_path)

    def choose_model(self):
        # open file dialog to select model path
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            filetypes=[("H5 Files", "*.h5")]
        )
        if file_path:
            self.model_path_label.config(text=file_path)

    def train_model(self):
        # get data directory and model path
        data_dir = self.data_path_label.cget("text")
        model_path = self.model_path_label.cget("text")
        if not data_dir:
            messagebox.showerror("Error", "Please choose a data directory.")
            return
        if not model_path:
            messagebox.showerror("Error", "Please choose a model path.")
            return
        # start train thread
        self.train_thread = TrainThread(model_path, data_dir)
        self.train_thread.run()


if __name__ == '__main__':
    root = tk.Tk()
    font.nametofont("TkDefaultFont").configure(
        size=12
    )
    window = MainWindow(root)
    root.mainloop()
