import re
import numpy as np
import utils
import constants
from collections import defaultdict, Counter


def get_agent_msgs(data):
    """
    Returns messages that were not sent by the customer.
    :param data: Raw conversations
    :return: List of sentences (split by '.' and '?')
    """
    sentences = []
    for entry in data['Issues']:
        for m in entry['Messages']:
            if not m['IsFromCustomer']:
                temp = m['Text'].replace('?', '.') #TODO- Include Exclamation??
                temp = temp.lower()
                sents = temp.split(".")
                for s in sents:
                    sentences.append(constants.START_TOKEN + ' ' + s + ' ' + constants.END_TOKEN)
                    #Append Start and end token to all sentences
    return sentences


def build_vocab(sentences):
    """
    Builds a set of unique words in the dataset
    :param sentences: List of sentences
    :return: vocab (python set() )
    """
    vocab = set()
    for i in range(len(sentences)):
        for w in utils.tokenize(sentences[i]):
            vocab.add(w)
    return vocab


def build_word_index(vocab):
    word_to_idx =