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
    sentence_count = Counter()
    sentence_list = list()
    for entry in data['Issues']:
        for m in entry['Messages']:
            if not m['IsFromCustomer']:
                temp = m['Text'].replace('?', '.') #TODO- Include Exclamation??
                temp = temp.lower()
                sents = temp.split(".")
                for s in sents:
                    sentence_count[(constants.START_TOKEN + ' ' + s.strip() + ' ' + constants.END_TOKEN)] += 1
                    sentence_list.append(constants.START_TOKEN + ' ' + s.strip() + ' ' + constants.END_TOKEN)
                    #Append Start and end token to all sentences
    return sentence_list, sentence_count



def build_inverted_index(sentences):
    """
    Builds an index of word to sentences
    :param sentences: List of unique sentences
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
    :param sentences: List of unique sentences
    :return: Index of first word to sentences
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
    sentence_list, sentence_counts = get_agent_msgs(data)

    ordered_unique_sents = []
    for s in sentence_counts.most_common():
        ordered_unique_sents.append(s[0])
    train_sentences = ordered_unique_sents[:-500]
    test_sentences = ordered_unique_sents[500:]
    for i in ordered_unique_sents:
        if "hey" in i:
            print(i)
    inverted_index = build_inverted_index(ordered_unique_sents)
    print(inverted_index['hey'])

    firstword_index = build_firstword_index(ordered_unique_sents)
    vocab = build_vocab(ordered_unique_sents)
    paragraph = ""
    for s in sentence_list:
        paragraph = paragraph + s + ' '

    utils.write_data_pkl(firstword_index, constants.FIRSTWORD_INDEX_FILE)
    utils.write_data_pkl(inverted_index, constants.INVERTED_INDEX_FILE)
    utils.write_data_pkl(vocab, constants.VOCAB_FILE)
    utils.write_data_pkl(paragraph, constants.PARAGRAPH_FILE)
    utils.write_data_pkl(ordered_unique_sents, constants.UNIQUE_SENTENCE_LIST_FILE)


if __name__ == '__main__':
    main()