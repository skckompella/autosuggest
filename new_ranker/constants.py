
RAW_DATA_FILE = "../data/sample_conversations.json"

INVERTED_INDEX_FILE = "../data/inverted_index.pkl"
FIRSTWORD_INDEX_FILE = "../data/firstword_index.pkl"
VOCAB_FILE = "../data/vocab.pkl"
UNIQUE_SENTENCE_LIST_FILE = "../data/sentence_list.pkl"
PARAGRAPH_FILE = "../data/paragraph.pkl"

CONTEXT_TRAIN_FILE = "../data/context_train.pkl"
CONTEXT_TEST_FILE = "../data/context_test.pkl"
RESPONSE_TRAIN_FILE = "../data/response_train.pkl"
RESPONSE_TEST_FILE = "../data/response_test.pkl"
LABEL_TRAIN_FILE = "../data/label_train.pkl"
LABEL_TEST_FILE = "../data/label_test.pkl"
WORD_INDEX_FILE = "../data/word_index.pkl"

WORD_MODEL_FILE = "../model/word_model.pkl"
BIGRAM_MODEL_FILE = "../model/bigram_model.pkl"

NUM_DISTRACTORS = 3

TRAIN_TEST_SPLIT = 0.8

START_TOKEN = '__start__'
END_TOKEN = '__end__'
UNK_TOKEN = '__unk__'
END_QUERY_TOKEN = '__endq__'
PAD_TOKEN = '__pad__'
PAD_IDX = 0

EMBEDDING_SIZE = 128
NUM_RNN_LAYERS = 1
RNN_HIDDEN_SIZE = 128
RNN_DROPOUT = 0.3

ONGPU = False
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 5

DEV_BATCH_SIZE = 512