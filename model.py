import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def standard_cnn(opt):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=[opt.image_width, opt.image_height, opt.num_channels]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    for i in range(1, opt.num_conv_layers):
        model.add(Conv2D((i+1) * 32, (5, 5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    for i in range(1, opt.num_fully_layers):
        model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model

def load_from_path(path):
    return load_model(path)

def save_graph(name):
    sess = tf.keras.backend.get_session()
    tf.train.write_graph(sess.graph, ".", name)
