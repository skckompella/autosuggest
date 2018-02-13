import json
import pickle as pkl
import re

def read_data_json(data_file):
    """
    Read from json file
    :param data_file: Pile path
    :return: data
    """
    with open(data_file) as json_data:
        data = json.load(json_data)
    return data


def write_data_pkl(data, data_file):
    """
    Write to pickle file
    :param data: Data
    :param data_file: File path
    :return:
    """
    with open(data_file, "wb") as fp:
        pkl.dump(data, fp)
    print("Saved file " + data_file)


def read_data_pkl(data_file):
    """
    Read from pickle file
    :param data_file: file path
    :return: data
    """
    with open(data_file, "rb") as fp:
        data = pkl.load(fp)
    return data


def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*\_]+|[^a-zA-Z\d\s]+", sentence)


def get_max_len(sentences):
    max_len = 0
    for s in sentences:
        max_len = max([max_len, len(s)])
    return max_len

#http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
#https://github.com/rrenaud/Gibberish-Detector/blob/master/gib_detect_train.py#L16
def get_chunks(l, n):
    for i in range(0, len(l) - n + 1):
        yield l[i:i+n]


def get_accuracy(predictions, labels):
    """
    Returns the accuracy of the predictions, provided the labels
    :param predictions: predictions from the model
    :param labels: gold labels for the respective predictions
    :return:
    """

    return float(sum(predictions == labels).data[0]) / labels.size()[0]