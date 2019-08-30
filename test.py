import config
import os
from model import load_from_path
from Dataset import Dataset, DATA_TEST

def test():
    if not config.LOAD_MODEL:
        raise RuntimeError("Train a model first")
    if not os.path.isfile(config.MODEL_PATH):
        raise RuntimeError("Model file not found")
    
    print ("Loading model")
    model = load_from_path(config.MODEL_PATH)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Metrics: " + str(model.metrics_names))
    
    print ("Loading testing data")
    dataset = Dataset(data_mode=DATA_TEST)
    (images, labels) = dataset.get_test_data()

    print ("Testing...")
    loss = model.evaluate(images, labels)
    print("loss: " + str(loss))
    print("Done")

if __name__ == "__main__":
    test()
