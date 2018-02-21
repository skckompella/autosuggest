import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import utils, constants

class TrainData(Dataset):
    """
    Uses Pytorch Dataset class to handle batching
    """
    def __init__(self):
        """

        """
        self.c_train = utils.read_data_pkl(constants.CONTEXT_TRAIN_FILE)
        self.r_train = utils.read_data_pkl(constants.RESPONSE_TRAIN_FILE)
        self.y_train = utils.read_data_pkl(constants.LABEL_TRAIN_FILE)
        self.train_len = self.c_train.shape[0] - 1
        shuffled_indices = np.random.randint(0, self.train_len, size=self.train_len)
        self.c_train = self.c_train[shuffled_indices]
        self.r_train = self.r_train[shuffled_indices]
        self.y_train = self.y_train[shuffled_indices]

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        return self.c_train[idx], self.r_train[idx], self.y_train[idx]


class DevData(Dataset):
    def __init__(self):
        self.c_dev = utils.read_data_pkl(constants.CONTEXT_TEST_FILE)
        self.r_dev = utils.read_data_pkl(constants.RESPONSE_TEST_FILE)
        self.y_dev = utils.read_data_pkl(constants.LABEL_TEST_FILE)
        self.dev_len = self.c_dev.shape[0] - 1

    def __len__(self):
        return self.dev_len

    def __getitem__(self, idx):
        return self.c_dev[idx], self.r_dev[idx], self.y_dev[idx]



class DualLstmRanker(nn.Module):
    def __init__(self, vocab_len, use_gru=False):
        super(DualLstmRanker, self).__init__()
        self.embedding_size = constants.EMBEDDING_SIZE
        self.vocab_len = vocab_len
        self.use_gru = use_gru

        self.embed = nn.Embedding(vocab_len, self.embedding_size, padding_idx=constants.PAD_IDX)
        if not use_gru:
            self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=constants.RNN_HIDDEN_SIZE,
                               num_layers=constants.NUM_RNN_LAYERS, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=constants.RNN_HIDDEN_SIZE,
                               num_layers=constants.NUM_RNN_LAYERS, batch_first=True)

        self.fc = nn.Linear(constants.RNN_HIDDEN_SIZE*constants.NUM_RNN_LAYERS,
                            constants.RNN_HIDDEN_SIZE*constants.NUM_RNN_LAYERS)

    def forward(self, c, r, predict=False):
        """

        :param c:
        :param r:
        :param mode:
        :return:
        """
        if not predict:
            c_embed = self.embed(c)
            r_embed = self.embed(r)

            if not self.use_gru:
                _, c_states = self.rnn(c_embed)
                _, r_states = self.rnn(r_embed)
                c_encoded = c_states[0]
                r_encoded = r_states[0]
            else:
                _, c_hidden = self.rnn(c_embed)
                _, r_hidden = self.rnn(r_embed)
                c_encoded = c_hidden
                r_encoded = r_hidden

            c_encoded = c_encoded.permute(1,0,2)
            r_encoded = r_encoded.permute(1,0,2)
            generated_r = self.fc(c_encoded.view(c_encoded.size(0), -1))
            logits = torch.bmm(generated_r.unsqueeze(1), r_encoded.view(r_encoded.size(0), -1).unsqueeze(2))
            logits = logits.squeeze(2)

            return logits

        else:
            pass







if __name__ == '__main__':
    pass





