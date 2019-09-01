import tensorflow as tf
import os
import sys
from model import standard_cnn, save_graph
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
    for epoch in range(opt.num_epochs):
        print("Training epoch {0} / {1}".format(epoch+1, opt.num_epochs))

        for images, labels in dataset.batch_iterator(): #Train model in steps to avoid loading all images to memory
            model.fit(images, labels, batch_size=opt.batch_size, epochs=1, callbacks=[tensorboard_callback])
            model.save(os.path.join(opt.model_dir, "cnn_epoch_{0}.hf".format(epoch)))
        model.save(os.path.join(opt.model_dir, "cnn_epoch_{0}.h5".format(epoch)))
    print(model.summary())
    print ("Model trained and saved to " + opt.model_dir)

if __name__ == "__main__":
    train()
