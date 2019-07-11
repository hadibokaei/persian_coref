import logging
import sys
from common import config

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('logs.txt', mode='a')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger(__name__)


def convert_to_numpy_array(input_file_name, output_file_name, vocab):
    logger.info("start to convert the file numpy arrays: {}".format(input_file_name))
    doc_word = []
    doc_char = []
    with open(input_file_name, 'r') as f:
        current_sentence_word = []
        current_sentence_char = []
        for line in f.readlines():
            splitted_line = line.split()
            if len(splitted_line) == 0:
                doc_word.append(current_sentence_word)
                doc_char.append(current_sentence_char)
            else:
                word = splitted_line[0]
                current_sentence_word.append(vocab.get_word_id(word, config.vocab_min_count))
                current_word_chars = [vocab.get_char_id(x) for x in word]
                current_word_chars = current_word_chars[:min(config.word_max_size,len(current_word_chars))]
                current_word_chars += [0] * (config.word_max_size - len(current_word_chars))
                current_sentence_char.append(current_word_chars)







