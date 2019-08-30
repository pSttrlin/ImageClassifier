#Model options
NUM_CLASSES = 2
NUM_CONV_LAYERS = 2
NUM_FULLY_LAYERS = 1
LOAD_MODEL = True
MODEL_PATH = "models/test.model"

#Training options
LR = 0.001
NUM_EPOCHS = 3
TENSORBOARD_LOGDIR = "logs"

#Dataset options
IMAGE_SIZE = (1000, 1500)
BATCH_SIZE = 8
NUM_CHANNELS = 1
SHUFFLE = True
TESTING_PATH = "data/test"
TRAINING_PATH = "data/train"
IMAGES_PER_STEP = 250 #Load 250 images to memory to train

#Constants
MODEL_DIR = "models/"
AD = True
NOAD = False
