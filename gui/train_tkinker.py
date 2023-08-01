from threading import Thread
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from commons.model import define_model

NUM_EPOCHS = 3
CLASS1_NAME = "cat"
CLASS2_NAME = "dog"


class UpdateWidgetsCallback(Callback):
    def __init__(self, progress_bar, result_text):
        super().__init__()
        self.progress_bar = progress_bar
        self.result_text = result_text

    def on_train_begin(self, logs=None):
        if logs is not None:
            self.progress_bar["maximum"] = self.params["steps"] * self.params["epochs"]
            self.progress_bar["value"] = 0
            self.result_text.insert(END, 'Training started...\n')

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            self.progress_bar["value"] = batch + 1

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            message = f'Epoch {epoch + 1}/{self.params["epochs"]}: loss={logs["loss"]:.4f}, acc={logs["accuracy"]:.4f}'
            self.progress_bar["value"] += self.params["steps"] * (epoch + 1)
            self.result_text.insert(END, message + "\n")


class TrainThread(Thread):
    def __init__(self, model_path, data_dir, update_widgets, result_text):
        super().__init__()
        self.result_text = result_text
        self.model_path = model_path
        self.data_dir = data_dir
        self.update_widgets = update_widgets

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

        # check files in directory
        if len(train_it.filenames) == 0:
            messagebox.showerror("Error", "No images found in directory.")
            return

        # fit model
        training = model.fit(train_it, steps_per_epoch=len(train_it),
                             epochs=NUM_EPOCHS, verbose=0,
                             callbacks=[self.update_widgets])

        # save model
        model.save(self.model_path)
        # get final accuracy and loss
        final_acc = training.history['accuracy'][-1]
        final_loss = training.history['loss'][-1]
        self.result_text.insert(END,
                                "Training Result\n" + f"Training finished. Accuracy: {final_acc:.5f}, Loss: {final_loss:.5f}")


class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title(f"Train {CLASS1_NAME.capitalize()} and {CLASS2_NAME.capitalize()}")
        self.train_thread = None
        # self.master.resizable(False, False)

        # create widgets
        self.title_label = Label(master, text=f"{CLASS1_NAME.capitalize()} vs {CLASS2_NAME.capitalize()} Trainer")
        self.title_label.config(font=("Arial", 16))
        self.title_label.pack(pady=10)

        self.data_frame = Frame(master)
        self.data_frame.pack(pady=10)

        self.data_path_frame = Frame(self.data_frame)
        self.data_path_frame.pack(side=BOTTOM)

        self.data_label = Label(self.data_frame, text="Data Directory:")
        self.data_label.pack(side=LEFT, padx=10)

        self.data_path_label = Label(self.data_path_frame, text="")
        self.data_path_label.pack(side=LEFT)

        self.data_button = Button(self.data_frame, text="Choose", command=self.choose_data)
        self.data_button.pack(side=LEFT, padx=10)

        self.model_frame = Frame(master)
        self.model_frame.pack(pady=10)

        self.model_path_frame = Frame(self.model_frame)
        self.model_path_frame.pack(side=BOTTOM)

        self.model_label = Label(self.model_frame, text="Model Path:")
        self.model_label.pack(side=LEFT, padx=10)

        self.model_path_label = Label(self.model_path_frame, text="")
        self.model_path_label.pack(side=LEFT)

        self.model_button = Button(self.model_frame, text="Choose", command=self.choose_model)
        self.model_button.pack(side=LEFT, padx=10)

        self.train_button = Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.stop_button = Button(master, text="Stop Training", command=self.stop_training, state=DISABLED)
        self.stop_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(master, orient=HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.result_label = Label(master, text="Training Result:")
        self.result_label.pack(pady=10)

        self.result_text = Text(master, height=10, width=50)
        self.result_text.pack()

        self.update_widgets = UpdateWidgetsCallback(self.progress_bar, self.result_text)

    def choose_data(self):
        # open file dialog to select data directory
        dir_path = filedialog.askdirectory(title="Open Data Directory")
        if dir_path:
            self.data_path_label.config(text=dir_path)

    def choose_model(self):
        # open file dialog to select model path
        file_path = filedialog.asksaveasfilename(title="Save Model", filetypes=[("H5 Files", "*.h5")])
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

        print(data_dir)



        # start train thread
        self.train_thread = TrainThread(model_path, data_dir, self.update_widgets, self.result_text)
        self.train_thread.start()
        self.train_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)

    def stop_training(self):
        try:
            # stop train thread
            self.train_thread.stop()
        except AttributeError:
            pass
        self.result_text.insert(END, "Training stopped.\n")
        # enable train button and disable stop button
        self.train_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)


if __name__ == '__main__':
    root = Tk()
    window = MainWindow(root)
    root.mainloop()
