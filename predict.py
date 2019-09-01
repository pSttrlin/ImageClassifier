import os
from model import load_from_path
from Dataset import Dataset, DATA_TEST
from ConfigParser import Parser

def predict(model, image_data):
    return model.predict(image_data)

if __name__ == "__main__":
    opt = Parser().parse()

    print("Loading model")
    model = load_from_path(opt.model_path)

    dataset = Dataset(opt, data_mode=DATA_TEST)
    totalPredictions = 0
    correctPredictions = 0
    for sample in dataset:
        prediction = round(predict(model, sample["data"])[0][0]) == 1
        print ("Predicted file {0}: {1} Label: {2}".format(sample["file"], prediction, sample["label"]))
        totalPredictions += 1
        if prediction == sample["label"]:
            correctPredictions += 1

    print ("Predicted {0} files, correct predictions={1}, accuracy={2}".format(totalPredictions,correctPredictions, correctPredictions / totalPredictions))
