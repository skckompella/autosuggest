import re
# import numpy as np
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


def build_inverted_index(sentences):
    """
    Builds an index of word to sentences
    :param sentences: List of sentences
    :return: Index of word to sentences
    """
    index = defaultdict(list)
    for i in range(len(sentences)):
        for w in utils.tokenize(sentences[i]):
            index[w].append(i)
    return index


def build_firstword_index(sentences):
    """
    Build index of first word of all sentences (excluding the start token)
    :param sentences:
    :return:
    """
    index = defaultdict(list)
    for i in range(len(sentences)):
        tokens = utils.tokenize(sentences[i])
        index[tokens[1]].append(i) #Excluding start tokens
    return index


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


def main():
    data = utils.read_data_json(constants.RAW_DATA_FILE)
    sentences = get_agent_msgs(data)
    # train_sentences =
    inverted_index = build_inverted_index(sentences)
    firstword_index = build_firstword_index(sentences)
    vocab = build_vocab(sentences)
    paragraph = ''
    for s in sentences:
        paragraph = paragraph + s + ' '

    utils.write_data_pkl(firstword_index, constants.FIRSTWORD_INDEX_FILE)
    utils.write_data_pkl(inverted_index, constants.INVERTED_INDEX_FILE)
    utils.write_data_pkl(vocab, constants.VOCAB_FILE)
    utils.write_data_pkl(paragraph, constants.PARAGRAPH_FILE)
    utils.write_data_pkl(sentences, constants.SENTENCES_LIST_FILE)

    # print data['Issues'][5]['Messages'][0]


if __name__ == '__main__':
    main()