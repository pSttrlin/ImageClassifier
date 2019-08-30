import tensorflow as tf
import numpy as np
import glob
import os
import config
import random
import cv2

DATA_TRAIN = 0
DATA_TEST = 1
DATA_BOTH = 2
class Dataset():

    training_files = []
    testing_files = []
    num_batches = 0

    def __init__(self, data_mode = DATA_BOTH):
        
        if (data_mode == DATA_BOTH or data_mode == DATA_TRAIN) and not os.path.isdir(config.TRAINING_PATH):
            raise RuntimeError("Directory {} not found".format(config.TRAINING_PATH))

        if (data_mode == DATA_BOTH or data_mode == DATA_TEST) and not os.path.isdir(config.TESTING_PATH):
            raise RuntimeError("Directory {} not found".format(config.TESTING_PATH))
        training_ads = 0
        testing_ads = 0
        
        if data_mode == DATA_BOTH or data_mode == DATA_TRAIN:
            for dir in [x[0] for x in os.walk(os.path.join(config.TRAINING_PATH, "Ads"))]:                
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.training_files.append((file, config.AD))
                    training_ads += 1

            for dir in [x[0] for x in os.walk(os.path.join(config.TRAINING_PATH, "Other"))]:     
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.training_files.append((file, config.NOAD))

        if data_mode == DATA_BOTH or data_mode == DATA_TEST:
            for dir in [x[0] for x in os.walk(os.path.join(config.TESTING_PATH, "Ads"))]:                
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.testing_files.append((file, config.AD))
                    testing_ads += 1

            for dir in [x[0] for x in os.walk(os.path.join(config.TESTING_PATH, "Other"))]:     
                for file in glob.glob(os.path.join(dir, "*.jpeg")):
                    self.testing_files.append((file, config.NOAD))

        print ("Loaded dataset with {0} training images ( {1} Ads and {2} Other ) and {3} testing images ( {4} Ads and {5} other )".format(
                                                                                                                                            len(self.training_files),
                                                                                                                                            training_ads,
                                                                                                                                            len(self.training_files) - training_ads,
                                                                                                                                            len(self.testing_files),
                                                                                                                                            testing_ads,
                                                                                                                                            len(self.testing_files) - testing_ads))
        self.num_batches = int(len(self.training_files) / config.BATCH_SIZE)
        random.shuffle(self.training_files)
        random.shuffle(self.testing_files)

    def batch_iterator(self):
        return self.BatchIterator(self)
    
    def read_image(self, image):
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    def process_image(self, img):
        img_res = cv2.resize(img, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        return img_res / 255.0

    def __iter__(self):
        for _ in range(len(self.testing_files)):
            yield self.get_sample()
    
    def get_sample(self):
        (image_file, label) = self.testing_files.pop()
        image_data = self.read_image(image_file)
        image_data = self.process_image(image_data)
        image_data = image_data.reshape(1, 1000, 1500, 1)
        return {
            "file": image_file,
            "data": image_data,
            "label": label
            }

    def get_batch(self):
        images = []
        labels = []
        for index in range(config.BATCH_SIZE):
            img, label = self.training_files.pop()
            img = self.read_image(img)
            img = self.process_image(img)
            images.append(img)
            labels.append(label)
        images = np.asarray(images)
        images = images.reshape(config.BATCH_SIZE, 1000, 1500, 1)
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
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            num_batches = int(len(self.dataset.training_files) / config.IMAGES_PER_STEP)
            for _ in range(num_batches):
                yield self.dataset.get_training_data(config.IMAGES_PER_STEP)
            lastBatch = self.dataset.get_training_data(len(self.dataset.training_files))
            yield lastBatch
