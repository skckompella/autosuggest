import json
import pickle as pkl
import re
import constants

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
    """
    Tokenize sentences
    :param sentence: String
    :return:
    """
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*\_]+|[^a-zA-Z\d\s]+", sentence)


def get_max_len(sentences):
    """
    Returns max length of sentences
    :param sentences: List of strings
    :return:
    """
    max_len = 0
    for s in sentences:
        max_len = max([max_len, len(tokenize(s))])
    return max_len


def get_chunks(l, n):
    """
    Returns a generator to generate chunks of length n
    :param l: Text
    :param n: chunk size
    :return: generator
    """
    for i in range(0, len(l) - n + 1):
        yield l[i:i+n]

def add_start_end(sentence):
    """
    Add start and end token as defined in constants
    :param sentence:
    :return:
    """
    return constants.START_TOKEN + ' ' + sentence.strip() + ' ' + constants.END_TOKEN

def remove_start_end(sentence):
    """
    Remove start and end tokes
    :param sentence:
    :return:
    """
    return sentence.replace(constants.START_TOKEN+" ", "").replace(" "+constants.END_TOKEN, "")

def get_accuracy(predictions, labels):
    """
    Returns the accuracy of the predictions, provided the labels
    :param predictions: predictions from the model
    :param labels: gold labels for the respective predictions
    :return:
    """

    return float(sum(predictions == labels).data[0]) / labels.size()[0]

def evaluate_recall(y_preds, y_true, k=3):
    num_examples = float(len(y_true))
    num_correct = 0
    for predictions, label in zip(y_preds, y_true):
        print(predictions)

        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples