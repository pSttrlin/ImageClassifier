import tensorflow as tf
import config
import os
from model import standard_cnn, save_graph
from Dataset import Dataset

def train():
    print("Loading dataset")
    dataset = Dataset()

    print("Generating model")
    model = standard_cnn()

    print("Preparing for training...")
    #(images, labels) = dataset.get_train_data() #Not efficent, loads all training data into memory
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print("Starting training")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_LOGDIR,
                                                          histogram_freq=0,
                                                          write_graph=True,
                                                          write_images=True,
                                                          update_freq="batch")
    for epoch in range(config.NUM_EPOCHS):
        print("Training epoch {0} / {1}".format(epoch+1, config.NUM_EPOCHS))

        for images, labels in dataset.batch_iterator(): #Train model in steps to avoid loading all images to memory
            model.fit(images, labels, batch_size=config.BATCH_SIZE, epochs=1, callbacks=[tensorboard_callback])
            model.save(os.path.join(config.MODEL_DIR, "cnn_epoch_{0}.hf".format(epoch)))
        model.save(os.path.join(config.MODEL_DIR, "cnn_epoch_{0}.h5".format(epoch)))
    print(model.summary())
    print ("Model trained and saved to " + config.MODEL_DIR)

if __name__ == "__main__":
    train()
