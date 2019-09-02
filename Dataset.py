import tensorflow as tf
import numpy as np
import glob
import os
import random
import cv2
from tensorflow.keras.utils import Sequence

DATA_TRAIN = 0
DATA_TEST = 1
DATA_BOTH = 2
AD = 1
NOAD = 0

class Dataset(Sequence):

    training_files = []
    testing_files = []
    num_batches = 0

    def __init__(self, opt, data_mode = DATA_BOTH):

        self.opt = opt

        if (data_mode == DATA_BOTH or data_mode == DATA_TRAIN) and not os.path.isdir(self.opt.training_path):
            raise RuntimeError("Directory {} not found".format(self.opt.training_path))

        if (data_mode == DATA_BOTH or data_mode == DATA_TEST) and not os.path.isdir(self.opt.testing_path):
            raise RuntimeError("Directory {} not found".format(self.opt.testing_path))
        training_ads = 0
        testing_ads = 0

        if data_mode == DATA_BOTH or data_mode == DATA_TRAIN:
            for dir in [x[0] for x in os.walk(os.path.join(self.opt.training_path, "Ads"))]:
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.training_files.append((file, AD))
                    training_ads += 1

            for dir in [x[0] for x in os.walk(os.path.join(self.opt.training_path, "Other"))]:
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.training_files.append((file, NOAD))

        if data_mode == DATA_BOTH or data_mode == DATA_TEST:
            for dir in [x[0] for x in os.walk(os.path.join(self.opt.testing_path, "Ads"))]:
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.testing_files.append((file, AD))
                    testing_ads += 1

            for dir in [x[0] for x in os.walk(os.path.join(self.opt.testing_path, "Other"))]:
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.testing_files.append((file, NOAD))

        print ("Loaded dataset with {0} training images ( {1} Ads and {2} Other ) and {3} testing images ( {4} Ads and {5} other )".format(
                                                                                                                                            len(self.training_files),
                                                                                                                                            training_ads,
                                                                                                                                            len(self.training_files) - training_ads,
                                                                                                                                            len(self.testing_files),
                                                                                                                                            testing_ads,
                                                                                                                                            len(self.testing_files) - testing_ads))
        self.num_batches = int(len(self.training_files) / self.opt.batch_size)
        random.shuffle(self.training_files)
        random.shuffle(self.testing_files)

    def __len__(self):
        return int(len(self.training_files) / self.opt.batch_size)

    def __getitem__(self, idx):
        length = (idx + 1) * self.opt.batch_size
        if len(self.training_files[idx * self.opt.batch_size:]) < self.opt.batch_size:
          length = len(self.training_files[idx * self.opt.batch_size:])

        batch_x = []
        batch_y = []
        for i in range(idx * self.opt.batch_size, length):
            image, label = self.training_files[i]
            image = self.read_image(image)
            image = self.process_image(image)
            batch_x.append(image)
            batch_y.append(label)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        return (batch_x, batch_y)


    def read_image(self, image):
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    def process_image(self, img):
        img_res = cv2.resize(img, (self.opt.image_width, self.opt.image_height))
        img_res = np.reshape(img_res, [self.opt.image_width, self.opt.image_height, self.opt.num_channels])
        return img_res / 255.0
