from model import CorefModel
from common.utility import get_all_files, logger, load_data
from common.vocabulary import Vocabulary
from os import listdir
from os.path import isdir, isfile, join
from common import config
import sys

lr = float(sys.argv[1])


data_files_path = []
data_files_path += get_all_files(config.path_data_train, '.npz')
for file_name in listdir(config.path_data_train):
    file_path = join(config.path_data_train, file_name)
    if isdir(file_path):
        data_files_path += get_all_files(file_path, '.npz')

num_files = len(data_files_path)
num_train_file = int(0.99*num_files)
num_validation_file = num_files - num_train_file
train_files_path = data_files_path[:num_train_file]
validation_files_path = data_files_path[num_train_file+1:]

logger.info("{} number of files found: {} train and {} validation".format(num_files, num_train_file, num_validation_file))

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
                   , lstm_unit_size=config.lstm_hidden_size, max_phrase_length=config.phrase_max_size, dir_tensoboard_log = config.path_tensorboard, dir_checkpoint=config.path_checkpoints)

model.build_graph()

try:
    file_name = model.restore_graph()
    splitted_file_name = file_name.split("-")
    epoch_number = int(splitted_file_name[-1]) + 1
    logger.info("the last checkpoint is loaded from {}. evaluation will be started...".format(file_name))
except Exception as e:
    logger.info("there is no checkpoint. first run the training script...")
    epoch_number = 0

model.train_phrase_identification(word_embedding, validation_files_path)


