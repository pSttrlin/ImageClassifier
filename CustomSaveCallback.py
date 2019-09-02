from tensorflow.keras.callbacks import Callback

class CustomSaveCallback(self, model, save_dir):

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.save_dir + "_" + str(epoch) + ".h5"
        self.model.save(save_path)
