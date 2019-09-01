import tensorflow as tf
import numpy as np
import glob
import os
import random
import cv2

DATA_TRAIN = 0
DATA_TEST = 1
DATA_BOTH = 2

class Dataset():

    training_files = []
    testing_files = []
    num_batches = 0
    AD = 1
    NOAD = 0

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

    def batch_iterator(self):
        return self.BatchIterator(self)

    def read_image(self, image):
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    def process_image(self, img):
        img_res = cv2.resize(img, (self.opt.image_width, self.opt.image_height))
        return img_res / 255.0

    def __iter__(self):
        for _ in range(len(self.testing_files)):
            yield self.get_sample()

    def get_sample(self):
        (image_file, label) = self.testing_files.pop()
        image_data = self.read_image(image_file)
        image_data = self.process_image(image_data)
        image_data = image_data.reshape(1, self.opt.image_width, self.opt.image_height, self.opt.num_channels)
        return {
            "file": image_file,
            "data": image_data,
            "label": label
            }

    def get_batch(self):
        images = []
        labels = []
        for index in range(self.opt.batch_size):
            img, label = self.training_files.pop()
            img = self.read_image(img)
            img = self.process_image(img)
            images.append(img)
            labels.append(label)
        images = np.asarray(images)
        images = images.reshape(self.opt.batch_size, self.opt.image_width, self.opt.image_height, self.opt.num_channels)
        labels = np.asarray(labels)
        return (images, labels)

    def get_train_data(self):
        images = []
        labels = []
        for img, label in self.training_files:
            img = self.read_image(img)
            img = self.process_image(img)
            images.append(img)
            labels.append(label)
            if len(images) > 300: break
        images = np.asarray(images)
        images = images.reshape(len(images), 1000, 1500, 1)
        labels = np.asarray(labels)
        return (images, labels)

    def get_training_data(self, size):
        images = []
        labels = []
        for _ in range(size):
            img, label = self.training_files.pop()
            img = self.read_image(img)
            img = self.process_image(img)
            images.append(img)
            labels.append(label)
        images = np.asarray(images)
        images = images.reshape(len(images), 1000, 1500, 1)
        labels = np.asarray(labels)
        return (images, labels)

    def get_test_data(self):
        images = []
        labels = []
        for img, label in self.testing_files:
            img = self.read_image(img)
            img = self.process_image(img)
            images.append(img)
            labels.append(label)
        images = np.asarray(images)
        images = images.reshape(len(images), 1000, 1500, 1)
        labels = np.asarray(labels)
        return (images, labels)

    class BatchIterator():
        def __init__(self, opt, dataset):
            self.dataset = dataset
            self.opt = opt

        def __iter__(self):
            num_batches = int(len(self.dataset.training_files) / self.self.opt.images_per_step)
            for _ in range(num_batches):
                yield self.dataset.get_training_data(self.self.opt.images_per_step)
            lastBatch = self.dataset.get_training_data(len(self.dataset.training_files))
            yield lastBatch
