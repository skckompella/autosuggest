"""
Reference: https://github.com/rodricios/autocomplete/
"""
import time
import re
# import numpy as np
import utils
import constants
from collections import defaultdict, Counter
from multiprocessing import Pool, Queue
from itertools import repeat

class BiGramModel():
    def __init__(self):
        self.start = constants.START_TOKEN
        self.end = constants.END_TOKEN
        self.unk = constants.UNK_TOKEN
        self.vocab = utils.read_data_pkl(constants.VOCAB_FILE)
        self.corpus = utils.read_data_pkl(constants.PARAGRAPH_FILE)
        self.inverted_index = utils.read_data_pkl(constants.INVERTED_INDEX_FILE)
        self.firstword_index = utils.read_data_pkl(constants.FIRSTWORD_INDEX_FILE)
        self.sentences = utils.read_data_pkl(constants.SENTENCES_LIST_FILE)


    def build_model(self, corpus):
        """
        Build bigram model
        :param corpus:
        :return:
        """
        words = utils.tokenize(corpus)
        self.word_model = Counter(words)           # Count(word)
        bigrams = list(utils.get_chunks(words, 2)) #Can be changed to any arbitrary ngrams
        self.bigram_model = defaultdict(Counter)   # Count(word2|word1)
        for tup in bigrams:
            try:
                self.bigram_model[tup[0]][tup[1]] += 1
            except:
                pass

    def save_model(self):
        utils.write_data_pkl(self.word_model, constants.WORD_MODEL_FILE)
        utils.write_data_pkl(self.bigram_model, constants.BIGRAM_MODEL_FILE)

    def load_model(self):
        self.word_model = utils.read_data_pkl(constants.WORD_MODEL_FILE)
        self.bigram_model = utils.read_data_pkl(constants.BIGRAM_MODEL_FILE)


    def cur_word(self, word, top_n=10):
        """
        Get the top_n most probable suggestions for a word with no preceeding words
        This function is never used because __START__ symbol is always prefixed
        :param word:
        :param top_n:
        :return:
        """
        try:
            return [w for w, c in self.word_model.most_common()
                    if w.startswith(word)][:top_n]
        except KeyError:
            raise Exception("No model available")


    def cur_given_prev(self, prev, cur, top_n=10):
        """
        Get the top_n most probable suggestions for a word given the preceeding word
        :param prev:
        :param cur:
        :param top_n:
        :return:
        """
        probable_words = [w for w, c in self.bigram_model[prev.lower()].most_common()
                          if w.startswith(cur)][:top_n]
        return probable_words

    def get_from_inverted_index(self, word):
        return set(self.inverted_index[word])

    def get_from_firstword_index(self, word):
        return set(self.firstword_index[word])

    def get_sent_match(self, idx, probable_prefix):
        for p in probable_prefix:
            if p in self.sentences[idx]:
                return self.sentences[idx]


    def predict(self, text, prev_predictions=None):
        """
        Predict probable sentences
        :param text: Text typed in so far
        :param prev_predictions: previous predictions for the same
        :return:
        """
        text = text.lower()
        parts = text.split()
        start_time = time.time()
        if len(parts) < 2:
            probable_words = self.cur_word(parts[0])
        else:
            probable_words = self.cur_given_prev(parts[-2], parts[-1])
        print("--- %s seconds ---" % (time.time() - start_time))


        probable_prefix = []
        predictions = []
        start_time = time.time()
        probable_sent_indices = self.get_from_inverted_index(parts[0])
        parts.pop(-1)
        prev_text = " ".join(parts)
        pool = Pool(processes=32)
        q = Queue()
        if prev_predictions is not None:
            pass
        else:
            for p in parts:
                probable_sent_indices.intersection(self.get_from_inverted_index(p))
            for p in probable_words:
                probable_sent_indices.intersection(self.get_from_inverted_index(p))
            for w in probable_words:
                probable_prefix.append(prev_text + ' ' + w)

            # with Pool(processes=32) as pool:
            #     sents = pool.starmap(self.get_sent_match, zip(probable_sent_indices, repeat(probable_prefix)))
            for s in probable_sent_indices:
                if any(x in self.sentences[s] for x in probable_prefix): #Using 'startswith' takes longer than 'in'
                    predictions.append(self.sentences[s]) #TODO- Sort by number of hits
        print("--- %s seconds ---" % (time.time() - start_time))

        return predictions


    def test(self):
        # self.build_model(self.corpus)
        self.load_model()
        print(self.predict(self.start + " hey"))
        print(self.predict(self.start + " I c"))
        print(self.predict(self.start + " It seems"))
        print(self.predict(self.start + " h"))

if __name__ == '__main__':
    model = BiGramModel()
    model.test()

