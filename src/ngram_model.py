
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
        self.corpus = utils.read_data_pkl(constants.PARAGRAPH_FILE)
        self.inverted_index = utils.read_data_pkl(constants.INVERTED_INDEX_FILE)
        self.firstword_index = utils.read_data_pkl(constants.FIRSTWORD_INDEX_FILE)
        self.sentences = utils.read_data_pkl(constants.UNIQUE_SENTENCE_LIST_FILE)


    def build_model(self, corpus):
        """
        Build bigram model
        :param corpus: Space separated string of all sentences
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
                print("Exception: Model not loaded?")

        self.save_model()

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
        :param word: Current word
        :param top_n: Top suggestions
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
        :param prev: previous word
        :param cur: Current word
        :param top_n: Number of top suggestions
        :return:
        """
        probable_words = [w for w, c in self.bigram_model[prev.lower()].most_common()
                          if w.startswith(cur)][:top_n]
        return probable_words

    def get_from_inverted_index(self, word):
        return set(self.inverted_index[word])

    def get_from_firstword_index(self, word):
        return set(self.firstword_index[word])

    def get_sent_match(self, idx, probable_prefix): #For multiprocess search. Unused currently
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
        text = self.start + ' ' + text
        parts = text.split()
        start_time = time.time()
        probable_words = self.cur_given_prev(parts[-2], parts[-1])
        print(" > Probable word prediction: %s seconds ---" % (time.time() - start_time))
        probable_prefix = []
        predictions = []
        start_time = time.time()
        probable_sent_indices = self.get_from_inverted_index(parts[0])
        parts.pop(-1)
        prev_text = " ".join(parts)
        # pool = Pool(processes=32)
        # q = Queue()
        if prev_predictions is not None:
            pass
        else:
            for p in parts:
                probable_sent_indices.intersection_update(self.get_from_inverted_index(p))
            temp = set()
            for p in probable_words:
                temp.update(probable_sent_indices.intersection(self.get_from_inverted_index(p)))
            probable_sent_indices.update(temp)

            for w in probable_words:
                probable_prefix.append(prev_text + ' ' + w)
            # with Pool(processes=32) as pool:
            #     sents = pool.starmap(self.get_sent_match, zip(probable_sent_indices, repeat(probable_prefix)))

            #NOTE- Using "in" is faster than startswith
            for s in sorted(probable_sent_indices):
                if any(self.sentences[s].startswith(x.strip()) for x in probable_prefix):
                    predictions.append(self.sentences[s].replace(self.start+" ", "").replace(" "+self.end, "")) #TODO- Sort by number of hits
        print(" > Probable sentence prediction: %s seconds ---" % (time.time() - start_time))

        return {"Suggestions": predictions}


    def test(self):
        self.build_model(self.corpus)
        start_time = time.time()
        # self.load_model()
        print(self.predict("I c"))
        print(" > Total time: %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        self.load_model()
        print(self.predict("It seems"))
        print(" > Total time: %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        self.load_model()
        print(self.predict("hey"))
        print(" > Total time: %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        self.load_model()
        print(self.predict("h"))
        print(" > Total time: %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        self.load_model()
        print(self.predict("i"))
        print(" > Total time: %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        self.load_model()
        print(self.predict("it seems we are"))
        print(" > Total time: %s seconds ---" % (time.time() - start_time))




if __name__ == '__main__':
    model = BiGramModel()
    model.test()


