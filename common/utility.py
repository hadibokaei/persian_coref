import logging
import sys
from common import config
import numpy as np

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
    '''
    تمام لیست‌های لازم را از داخل یک فایل آموزشی استخراج می‌کند و در قالب یک فایل ذخیره می‌کند. مواردی که ساتخراج می‌شود:
    doc_word: shape = [num of sentences, num of words in sentences] تعداد کلمات داخل جملات یکسان نیست
    doc_char: shape = [num of sentences, num of words in sentences, max num of chars in a word]
    :param input_file_name:
    :param output_file_name:
    :param vocab:
    :return:
    '''
    doc_word = []
    doc_word_raw = []
    doc_char = []
    with open(input_file_name, 'r') as f:
        current_sentence_word = []
        current_sentence_char = []
        current_sentence_word_raw = []
        for line in f.readlines():
            splitted_line = line.split()
            if len(splitted_line) == 0:
                doc_word.append(current_sentence_word)
                doc_word_raw.append(current_sentence_word_raw)
                doc_char.append(current_sentence_char)
                current_sentence_word = []
                current_sentence_char = []
                current_sentence_word_raw = []
            else:
                word = splitted_line[0]
                current_sentence_word.append(vocab.get_word_id(word, config.vocab_min_count))
                current_sentence_word_raw.append(line)
                current_word_chars = [vocab.get_char_id(x) for x in word]
                current_word_chars = current_word_chars[:min(config.word_max_size,len(current_word_chars))]
                current_word_chars += [0] * (config.word_max_size - len(current_word_chars))
                current_sentence_char.append(current_word_chars)
        if len(current_sentence_word) > 0:
            doc_word_raw.append(current_sentence_word_raw)
            doc_word.append(current_sentence_word)
            doc_char.append(current_sentence_char)
    phrase_word = []
    phrase_word_len = []
    gold_phrase = []
    for sentence_order in range(len(doc_word)):
        for start_word_order in range(len(doc_word[sentence_order])):
            for end_word_order in range(start_word_order + 1, len(doc_word[sentence_order]) + 1):
                if end_word_order - start_word_order > config.phrase_max_size:
                    break
                current_phrase = [[sentence_order + 1 , k] for k in range(start_word_order, end_word_order)]
                current_phrase_len = len(current_phrase)

                if current_phrase_len == 1:
                    single_word = doc_word_raw[current_phrase[0][0]-1][current_phrase[0][1]]
                    single_tag = single_word.split()[2]
                    if is_number(single_tag):
                        gold_phrase.append(1)
                    else:
                        gold_phrase.append(0)

                else:
                    start_word = doc_word_raw[current_phrase[0][0]-1][current_phrase[0][1]]
                    start_tag = start_word.split()[2]
                    end_word = doc_word_raw[current_phrase[-1][0]-1][current_phrase[-1][1]]
                    end_tag = end_word.split()[2]

                    all_other_word_is_middle = True
                    if start_tag.endswith("(*") and end_tag.endswith("*)"):
                        for w in doc_word_raw[current_phrase[0][0]-1][current_phrase[0][1]+1:current_phrase[-1][1]]:
                            other_tag = w.split()[2]
                            if other_tag != "*":
                                all_other_word_is_middle = False
                                break
                        if all_other_word_is_middle:
                            gold_phrase.append(1)
                        else:
                            gold_phrase.append(0)
                    else:
                        gold_phrase.append(0)


                current_phrase += [[0,0]] * (config.phrase_max_size - len(current_phrase))
                phrase_word.append(current_phrase)
                phrase_word_len.append(current_phrase_len)

    assert len(phrase_word_len) == len(phrase_word)
    assert len(phrase_word) == len(gold_phrase)

    # for i in range(len(gold_phrase)):
    #     if gold_phrase[i] == 1:
    #         phrase = doc_word_raw[phrase_word[i][0][0]-1][phrase_word[i][0][1]:phrase_word[i][phrase_word_len[i]-1][1]+1]
    #         print(' '.join([x.split()[0] for x in phrase]))

    np.savez_compressed(output_file_name, doc_word=doc_word
                        , doc_char=doc_char, phrase_word = phrase_word
                        , phrase_word_len=phrase_word_len, gold_phrase=gold_phrase)



def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False




