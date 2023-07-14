from tkinter import filedialog, messagebox, scrolledtext, END, TOP, HORIZONTAL, DISABLED, LEFT, NORMAL, Tk, font
from tkinter.ttk import Progressbar, Button, Label, Entry
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
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


class TrainThread:
    def __init__(self, model_path, data_dir, update_widgets):
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
        train_it = datagen.flow_from_directory(self.data_dir, class_mode='binary',
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
        self.update_widgets.result_text.insert(END, f"Training finished. Accuracy: {final_acc}, Loss: {final_loss}\n")


class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title(f"Train {CLASS1_NAME.capitalize()} and {CLASS2_NAME.capitalize()}")

        # create widgets
        self.title_label = Label(master, text=f"{CLASS1_NAME.capitalize()} vs {CLASS2_NAME.capitalize()} Trainer")
        self.title_label.pack(side=TOP, pady=10)

        self.data_label = Label(master, text="Data Directory:")
        self.data_label.pack(side=TOP, pady=5)
        self.data_path_label = Entry(master, width=50)
        self.data_path_label.pack(side=LEFT, padx=5)
        self.data_button = Button(master, text="Choose", command=self.choose_data)
        self.data_button.pack(side=LEFT, padx=5)

        self.model_label = Label(master, text="Model Path:")
        self.model_label.pack(side=TOP, pady=5)
        self.model_path_label = Entry(master, width=50)
        self.model_path_label.pack(side=LEFT, padx=5)
        self.model_button = Button(master, text="Choose", command=self.choose_model)
        self.model_button.pack(side=LEFT, padx=5)

        self.train_button = Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack(side=TOP, pady=10)
        self.stop_button = Button(master, text="Stop Training", command=self.stop_training)
        self.stop_button.pack(side=TOP, pady=5)
        self.stop_button["state"] = DISABLED

        self.progress_bar = Progressbar(master, orient=HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack(side=TOP, pady=10)

        self.result_label = Label(master, text="Training Result:")
        self.result_label.pack(side=TOP, pady=5)
        self.result_text = scrolledtext.ScrolledText(master, width=50, height=10)
        self.result_text.pack(side=TOP, pady=5)

        self.update_widgets = UpdateWidgetsCallback(self.progress_bar, self)

    def choose_data(self):
        # open file dialog to select data directory
        dir_path = filedialog.askdirectory(title="Open Data Directory")
        if dir_path:
            self.data_path_label.delete(0, END)
            self.data_path_label.insert(0, dir_path)

    def choose_model(self):
        # open file dialog to select model path
        file_path = filedialog.asksaveasfilename(title="Save Model", filetypes=[("H5 Files", "*.h5")])
        if file_path:
            self.model_path_label.delete(0, END)
            self.model_path_label.insert(0, file_path)

    def train_model(self):
        # get data directory and model path
        data_dir = self.data_path_label.get()
        model_path = self.model_path_label.get()
        if not data_dir:
            messagebox.showerror("Error", "Please choose a data directory.")
            return
        if not model_path:
            messagebox.showerror("Error", "Please choose a model path.")
            return
        # start train thread
        self.train_thread = TrainThread(model_path, data_dir, self.update_widgets)
        self.train_thread.run()
        self.train_button["state"] = DISABLED
        self.stop_button["state"] = NORMAL

    def stop_training(self):
        self.train_thread.update_widgets.result_text.insert(END, "Training stopped.\n")
        # enable train button and disable stop button
        self.train_button["state"] = NORMAL
        self.stop_button["state"] = DISABLED


if __name__ == '__main__':
    root = Tk()
    font.nametofont("TkDefaultFont").configure(
        size=12
    )
    window = MainWindow(root)
    root.mainloop()
