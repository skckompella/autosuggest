import re
import numpy as np
from collections import defaultdict, Counter

from src import utils
from src import constants


def get_msgs(data):
    """
    Get
    :param data:
    :return:
    """
    queries = []
    answers = []
    for entry in data['Issues']:
        aFlag = False
        cFlag = False
        q = ''
        for m in entry['Messages']:
            if m['IsFromCustomer']:
                cFlag = True
                q = q + m['Text'] + ' '
            else:
                sents = m['Text'].replace('?', '.').lower().split(".")
                if len(sents[0]) > 1 and len(q) > 0:
                    queries.append(q + ' ' + constants.END_QUERY_TOKEN)
                    answers.append(utils.add_start_end(sents[0]))
                for i in range(1, len(sents)):
                    if len(sents[i].strip()) > 0 and (len(q) > 0 or aFlag):
                    #Condition 1: Ignoring agent only messages. There is no context to be inferred from that
                    #Condition 2: Include msgs where multiple answers entries are present for a single query (using aFlag)
                        queries.append(queries[-1] + ' ' + utils.remove_start_end(sents[i-1]))
                        answers.append(utils.add_start_end(sents[i]))
                aFlag = True

    return queries, answers



def build_inverted_index(sentences):
    """
    Builds an index of word to sentences
    :param sentences: List of unique sentences
    :return: Index of word to sentences o
    """
    index = defaultdict(list)
    for i in range(len(sentences)):
        for w in utils.tokenize(sentences[i]):
            index[w].append(i)
    return index

def build_sent_index(sentences):
    """

    :param sentences:
    :return:
    """
    index = dict()
    for i in range(len(sentences)):
        index[sentences[i]] = i
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

    :param sentences:
    :return:
    """
    vocab = set()
    for i in range(len(sentences)):
        for w in utils.tokenize(sentences[i]):
            vocab.add(w)
    return vocab


def build_word_index(vocab):
    """
    Build word index and index to word
    :param vocab: Vocabulary
    :return: word index, index to word dicts
    """
    word_to_index = dict()
    index_to_word = dict()
    word_to_index[constants.PAD_IDX] = 0
    index_to_word[0] = constants.PAD_IDX
    for idx, w in enumerate(vocab):
        word_to_index[w] = idx+1 # accounting for Padding
        index_to_word[idx+1] = w

    return word_to_index, index_to_word


def get_unique_sentences(sentences):
    """
    Returns a list of sentences ordered by the number of hits
    :param sentences: Unordered sentences
    :return: List of sentences
    """
    ctr = Counter(sentences)
    return [s[0] for s in ctr.most_common()]


def build_data(queries, answers, ordered_unique_sents, firstword_index, answer_index):
    """
    Build Context-Response pairs and corresponding labels.
    Distractor responses are generated as follows:
     - Find sentences that have the same first word as that of the right answer
     - Add the top 9 most used sentences as the distractors
    :param queries: List of queries
    :param answers: list of answers
    :param firstword_index: Answers indexed by its first word
    :return:
    """
    contexts = []
    responses = []
    labels = []
    for i in range(len(queries)):
        contexts.append(queries[i])
        responses.append(answers[i])
        labels.append(1)

        distractors = firstword_index[utils.tokenize(answers[i])[1]][:constants.NUM_DISTRACTORS]
        if answer_index[answers[i]] in distractors:
            distractors.remove(answer_index[answers[i]])
        while len(distractors) < constants.NUM_DISTRACTORS:
            r = np.random.randint(0, len(ordered_unique_sents)-1)
            if r not in distractors and r != answer_index[answers[i]]:
                distractors.append(r)

        for j in distractors:
            contexts.append(queries[i])
            responses.append(ordered_unique_sents[j])

            labels.append(0)
    return contexts, responses, labels


def vectorize(sentences, word_to_index, max_len):
    """
    Vectorize uisng word indices
    :param sentences:
    :param word_to_index:
    :param max_len: Maximum sentence length
    :return:
    """
    vectors = []
    for i in range(len(sentences)):
        v = []
        for w in utils.tokenize(sentences[i]):
            v.append(word_to_index[w])
        vectors.append(v)
        for j in range(len(v), max_len):
            v.append(0)
    return np.asarray(vectors)


def main():
    data = utils.read_data_json(constants.RAW_DATA_FILE)
    queries, answers = get_msgs(data)

    #Order by number of hits
    ordered_unique_sents = get_unique_sentences(answers)

    #Combine ngram training data into a single paragraph
    paragraph = " ".join(ordered_unique_sents)

    #Build vocab
    vocab = build_vocab(queries + answers)
    print("Vocab size: ", len(vocab))
    #Build word indices
    word_to_index, index_to_word = build_word_index(vocab)

    # Build indexes for retrieval
    inverted_index = build_inverted_index(ordered_unique_sents)
    firstword_index = build_firstword_index(ordered_unique_sents)
    answer_index = build_sent_index(ordered_unique_sents)

    #Split
    q_train = queries[:int(constants.TRAIN_TEST_SPLIT*len(queries))]
    q_test = queries[int(constants.TRAIN_TEST_SPLIT*len(queries)):]
    a_train = answers[:int(constants.TRAIN_TEST_SPLIT*len(queries))]
    a_test = answers[int(constants.TRAIN_TEST_SPLIT*len(queries)):]

    #Build data
    contexts_train, responses_train, labels_train = build_data(q_train, a_train,
                                                               ordered_unique_sents, firstword_index, answer_index)
    contexts_test, responses_test, labels_test = build_data(q_test, a_test,
                                                            ordered_unique_sents, firstword_index, answer_index)


    # Tokenize and vectorize all data
    context_max_len = utils.get_max_len(contexts_train+contexts_test)
    resp_max_len = utils.get_max_len(responses_test+responses_train)
    max_len = max(context_max_len, resp_max_len)
    contexts_train = vectorize(contexts_train, word_to_index, max_len)
    contexts_test = vectorize(contexts_test, word_to_index, max_len)
    responses_train = vectorize(responses_train, word_to_index, max_len)
    responses_test = vectorize(responses_test, word_to_index, max_len)
    labels_train = np.asarray(labels_train)
    labels_test = np.asarray(labels_test)

    print("Contexts train shape: ", contexts_train.shape)
    print("Contexts test shape: ", contexts_test.shape)
    print("Responses train shape: ", responses_train.shape)
    print("Responses train shape: ", responses_test.shape)
    print("Labels train shape: ", labels_train.shape)
    print("Labels train shape: ", labels_test.shape)
    print(labels_train)

    #Write preprocessed data to files
    utils.write_data_pkl(contexts_train, constants.CONTEXT_TRAIN_FILE)
    utils.write_data_pkl(contexts_test, constants.CONTEXT_TEST_FILE)
    utils.write_data_pkl(responses_train, constants.RESPONSE_TRAIN_FILE)
    utils.write_data_pkl(responses_test, constants.RESPONSE_TEST_FILE)
    utils.write_data_pkl(labels_train, constants.LABEL_TRAIN_FILE)
    utils.write_data_pkl(labels_test, constants.LABEL_TEST_FILE)
    utils.write_data_pkl(word_to_index, constants.WORD_INDEX_FILE)
    utils.write_data_pkl(inverted_index, constants.INVERTED_INDEX_FILE)
    utils.write_data_pkl(firstword_index, constants.FIRSTWORD_INDEX_FILE)
    utils.write_data_pkl(paragraph, constants.PARAGRAPH_FILE)
    utils.write_data_pkl(ordered_unique_sents, constants.UNIQUE_SENTENCE_LIST_FILE)

if __name__ == '__main__':
    main()