import tensorflow as tf
import config
from collections import OrderedDict
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def standard_cnn():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), input_shape=[config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], config.NUM_CHANNELS]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    for i in range(1, config.NUM_CONV_LAYERS):
        model.add(Conv2D((i+1) * 32, (5, 5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    for i in range(1, config.NUM_FULLY_LAYERS):
        model.add(Dense(64))
    
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model

def load_from_path(path):
    return load_model(path)

def save_graph(name):
    sess = tf.keras.backend.get_session()
    tf.train.write_graph(sess.graph, ".", name)
