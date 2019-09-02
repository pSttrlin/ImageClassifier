import tensorflow as tf
import os
import sys
from CustomSaveCallback import CustomSaveCallback
from model import standard_cnn, visualize_layer
from Dataset import Dataset, DATA_TRAIN
from ConfigParser import Parser

def train():
    opt = Parser().parse()

    print("Loading dataset")
    dataset = Dataset(opt, data_mode=DATA_TRAIN)

    print("Generating model")
    model = standard_cnn(opt)

    print("Preparing for training...")
    #(images, labels) = dataset.get_train_data() #Not efficent, loads all training data into memory
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    print("Starting training")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=opt.tensorboard_logdir,
                                                          histogram_freq=0,
                                                          write_graph=True,
                                                          write_images=True,
                                                          update_freq="batch")
    save_callback = CustomSaveCallback(opt.model_dir + "cnn")
    model.fit_generator(generator=dataset,
                        steps_per_epoch=(len(dataset.training_files) // opt.batch_size),
                        epochs=opt.num_epochs,
                        verbose=1,
                        callbacks=[tensorboard_callback, save_callback])
    print(model.summary())
    print ("Model trained and saved to " + opt.model_dir)

if __name__ == "__main__":
    train()
