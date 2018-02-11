import re
import utils
import constants
from collections import defaultdict, Counter

def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*]+|[^a-zA-Z\d\s]+", sentence)


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
                sents = temp.split(".")
                for s in sents:
                    sentences.append(s)
    # print sentences[:300]
    return sentences
    # max_len = 0
    # print len(sentences)
    # for s in sentences:
    #     max_len = max([max_len, len(s)])
    # print max_len


def build_inverted_index(sentences):
    index = defaultdict(list)
    for i in range(len(sentences)):
        for w in tokenize(sentences[i]):
            index[w].append(i)
    print index
    return index


def build_vocab(sentences):
    vocab = set()


data = utils.read_data_json(constants.RAW_DATA_FILE)
sentences = get_agent_msgs(data)
index = build_inverted_index(sentences)

# print data['Issues'][5]['Messages'][0]