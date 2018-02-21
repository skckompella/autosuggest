from keras import Input
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Dropout, Flatten, LSTM, merge
from keras import backend as K

import numpy as np
import constants
import utils


class Rank():
    def __init__(self, vocab_len, max_len):
        self.c = Input(shape=(max_len,))
        self.r = Input(shape=(max_len,))

        self.embed = Embedding(input_dim=vocab_len, output_dim=constants.EMBEDDING_SIZE, input_length=max_len)
        self.c_embed = self.embed(self.c)
        self.r_embed = self.embed(self.r)

        self.rnn = LSTM(units=constants.RNN_HIDDEN_SIZE)
        self.c_enc = self.rnn(self.c_embed)
        self.r_enc = self.rnn(self.r_embed)

        self.gen = Dense(units=constants.EMBEDDING_SIZE)
        self.gen_r = self.gen(self.c_enc)

        self.sim = merge([self.gen_r, self.r_enc], mode='dot', dot_axes=1)
        self.sim = Activation('sigmoid')(self.sim)
        self.model = Model(inputs=[self.c, self.r], outputs=self.sim)
        print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    def fit(self, c_train, r_train, y_train, c_dev, r_dev, y_dev):
        self.model.fit([c_train, r_train], y_train,
                       batch_size=constants.BATCH_SIZE,
                       epochs=constants.NUM_EPOCHS,
                       verbose=1,
                       validation_data=([c_dev, r_dev], y_dev))

    def predict(self, c_test, r_test, y_test):
        probs = self.model.predict([c_test, r_test], batch_size=constants.BATCH_SIZE)
        print(probs[:10])


if __name__ == '__main__':
    c_train = utils.read_data_pkl(constants.CONTEXT_TRAIN_FILE)
    r_train = utils.read_data_pkl(constants.RESPONSE_TRAIN_FILE)
    y_train = utils.read_data_pkl(constants.LABEL_TRAIN_FILE)
    c_dev = utils.read_data_pkl(constants.CONTEXT_TEST_FILE)
    r_dev = utils.read_data_pkl(constants.RESPONSE_TEST_FILE)
    y_dev = utils.read_data_pkl(constants.LABEL_TEST_FILE)
    word_to_idx = utils.read_data_pkl(constants.WORD_INDEX_FILE)
    model = Rank(len(word_to_idx), c_train.shape[1])
    model.fit(c_train, r_train, y_train, c_dev, r_dev, y_dev)
    model.predict(c_dev, r_dev, y_dev)