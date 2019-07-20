from model import CorefModel
import tensorflow as tf
from common.utility import get_all_files, logger, load_data
from common.vocabulary import Vocabulary
from os import listdir
from os.path import isdir, isfile, join
from common import config
import numpy as np


data_files_path = []

data_files_path += get_all_files(config.path_data_train, '.npz')
for file_name in listdir(config.path_data_train):
    file_path = join(config.path_data_train, file_name)
    if isdir(file_path):
        data_files_path += get_all_files(file_path, '.npz')

logger.info("{} number of files found.".format(len(data_files_path)))

all_docs_word_ids = []
all_docs_char_ids = []
all_docs_phrase_indices = []
all_docs_gold_phrases = []
all_docs_phrase_length = []
all_docs_pair_indices = []
all_docs_pair_golds = []
logger.info("start to load the data files with format npz...")
# data_files_path = data_files_path[146:147]
for file in data_files_path:
    doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, pair_indices, pair_gold = load_data(file)
    if len(doc_word) == 0:
        print("skip this file (zero length document): {}".format(file))
        continue
    if np.sum(gold_phrase) == 0:
        print("skip this file (no phrase): {}".format(file))
        continue
    all_docs_word_ids.append(doc_word)
    all_docs_char_ids.append(doc_char)
    all_docs_phrase_indices.append((phrase_word))
    all_docs_phrase_length.append(phrase_word_len)
    all_docs_gold_phrases.append(gold_phrase)
    all_docs_pair_indices.append(pair_indices)
    all_docs_pair_golds.append(pair_gold)

vocab = Vocabulary()
vocab.load_vocab(config.path_vocabulary)


if isfile(config.path_trimmed_word_embedding + ".npz"):
    answer = input("trimmed word embedding exists. load(press 'l') or build(press anything other than 'l') again:")
    if answer == "l":
        word_embedding = vocab.load_trimmed_word_embeddings(config.path_trimmed_word_embedding)
    else:
        word_embedding = vocab.dump_trimmed_word_embeddings(we_file_name=config.path_full_word_embedding,
                                                            trimmed_file_name=config.path_trimmed_word_embedding,
                                                            dim=config.word_embedding_dimension)
else:
    word_embedding = vocab.dump_trimmed_word_embeddings(we_file_name=config.path_full_word_embedding,
                                                        trimmed_file_name=config.path_trimmed_word_embedding,
                                                        dim=config.word_embedding_dimension)

model = CorefModel(word_vocab_size=vocab.last_word_index + 1, char_vocab_size=vocab.char_size, word_embedding_dimension=config.word_embedding_dimension
                   , char_embedding_dimension=config.char_embedding_dimension, max_word_length=config.word_max_size, conv_filter_num=2, conv_filter_size=[2,3,4,5]
                   , lstm_unit_size=config.lstm_hidden_size, max_phrase_length=config.phrase_max_size, dir_tensoboard_log = config.path_tensorboard)

model.build_graph()
# model.train_pair_identification(word_embedding, all_docs_word_ids, all_docs_char_ids, all_docs_phrase_indices
#                                     , all_docs_gold_phrases, all_docs_phrase_length
#                                     , all_docs_pair_indices, all_docs_pair_golds
#                                     , epoch_start=0, max_epoch_number=config.max_epoch_number)

model.train_phrase_identification(word_embedding, all_docs_word_ids, all_docs_char_ids, all_docs_phrase_indices
                                    , all_docs_gold_phrases, all_docs_phrase_length
                                    , epoch_start=0, max_epoch_number=config.max_epoch_number)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# print(model.gold_phrases.eval())
# print(np.shape(model.gold_phrases.eval()))


