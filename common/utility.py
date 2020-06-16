import logging
import sys
from common import config
import numpy as np
from os import listdir
from os.path import isdir, isfile, join
import pickle

def setup_custom_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.StreamHandler()
        logger.addHandler(console)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        console.setFormatter(formatter)
    return logger

# def setup_custom_logger(name):
#     logger = logging.getLogger(name)
#     if not logger.handlers:
#         # formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
#         #                               datefmt='%Y-%m-%d %H:%M:%S')
#         handler = logging.FileHandler('logs.txt', mode='a')
#         # handler.setFormatter(formatter)
#         screen_handler = logging.StreamHandler()
#         # screen_handler.setFormatter(formatter)
#         logger.setLevel(logging.DEBUG)
#         logger.addHandler(handler)
#         logger.addHandler(screen_handler)
#     return logger

logger = setup_custom_logger(__name__)


def convert_to_numpy_array(input_file_name, output_file_name, vocab):
    '''
    تمام لیست‌های لازم را از داخل یک فایل آموزشی استخراج می‌کند و در قالب یک فایل ذخیره می‌کند. مواردی که ساتخراج می‌شود:
    doc_word:       shape = [num of sentences, num of words in sentences] تعداد کلمات داخل جملات یکسان نیست
    doc_char:       shape = [num of sentences, num of words in sentences, max num of chars in a word] به ازای هر جمله تعداد کلمات موجود در آن یکسان نیست. اما تعداد حروف داخل هر کلمه ثابت و به تعداد بیشینه طول کلمه تنظیم‌شده در فایل کانفیگ در نظر گرفته شده است
    phrase_word:    shape = [num of candidate phrases, max num of words in a phrase, 2] بعد سوم ۲ عدد است: عدد اول نشان‌دهنده ترتیب جمله است که از ۱ شروع می‌شود. عدد دوم نشان‌دهنده ترتیب کلمه در جمله است که برای هر جمله از ۰ شروع می‌شود
    phrase_word_len:shape = [num of candidate phrases] تعداد کلمات داخل هر عبارت را نشان می‌دهد. کلمات اضافی داخل عبارت با مقدار [۰و۰] پر شده است
    gold_phrase:    shape = [num of candidate phrases] یک لیست باینری است. به ازای هر عبارت کاندید مشخص می‌کند که آیا در پیکره به عنوان یک عبارت برچسب خورده است یا نه
    clusters:       کلاسترهایی را که توسط برچسب‌های مرجع در متن مشخص شده است را ذخیره می‌کند
    gold_2_local_phrase_id_map: یک دیکشنری است که کلید آن شناسه عبارت مشخص شده در متن مرجع است و کلید آن شناسه آن عبارت در لیست‌های بالا می‌باشد.
    pair_indices:   shape = [num of candidate pairs, 2, 1] یک لیست است که اندیس عباراتی را مشخص می‌کند که با هم می‌توانند جفت شوند
    pair_gold:      shape = [num of candidate pairs] مشخص می‌کند که آیا جفت عبارت کاندید در پیکره به عنوان هم‌مرجع برچسب خورده‌اند یا خیر
    :param input_file_name: فایل ورودی که قرار است اطلاعات آنم استخراج شود.
    :param output_file_name: فایل خروجی که قرار است لیست‌های تولید شده به صورت فشرده روی آن ذخیره شود
    :param vocab: واژگان استخراج شده
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
                assert len(current_sentence_word) == len(current_sentence_char)
                if len(current_sentence_word) == 0:
                    print("===========================================================================")
                    continue
                assert len(current_sentence_word) > 0
                assert len(current_sentence_char) > 0
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
    phrase_id = []
    phrase_id_pair = []
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
                        current_id = int(single_tag)
                        prev_id = int(single_word.split()[3])
                        phrase_id.append(current_id)
                        phrase_id_pair.append([current_id, prev_id])
                        gold_phrase.append(1)
                    else:
                        phrase_id.append(-1)
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
                            current_id = int(start_word.split()[2].replace("(*",""))
                            prev_id = int(start_word.split()[3].replace("(*",""))
                            phrase_id.append(current_id)
                            phrase_id_pair.append([current_id, prev_id])
                            gold_phrase.append(1)
                        else:
                            phrase_id.append(-1)
                            gold_phrase.append(0)
                    else:
                        phrase_id.append(-1)
                        gold_phrase.append(0)


                current_phrase += [[0,0]] * (config.phrase_max_size - len(current_phrase))
                phrase_word.append(current_phrase)
                phrase_word_len.append(current_phrase_len)

    assert len(phrase_word_len) == len(phrase_word)
    assert len(phrase_word) == len(gold_phrase)
    assert len(phrase_word) == len(phrase_id)

    phrase_id_np = np.array(phrase_id)
    local_phrase_ids = np.squeeze(np.argwhere(phrase_id_np > 0))
    gold_phrase_ids = phrase_id_np[local_phrase_ids]
    gold_2_local_phrase_id_map = dict(zip(gold_phrase_ids, local_phrase_ids))


    clusters = []
    for pair in phrase_id_pair:
        found = False
        for i in range(len(clusters)):
            if pair[0] in clusters[i] or pair[1] in clusters[i]:
                clusters[i].add(pair[0])
                clusters[i].add(pair[1])
                found = True
        if not found:
            clusters.append(set())
            clusters[-1].add(pair[0])
            clusters[-1].add(pair[1])

    pickle.dump([doc_word,doc_char,phrase_word,phrase_word_len,gold_phrase,clusters,gold_2_local_phrase_id_map], open(output_file_name, "wb"))


def pair_has_overlap(phrase1, phrase2):
    has_overlap = False
    for i in range(len(phrase1)):
        for j in range(len(phrase2)):
            if phrase1[i][0] == phrase2[j][0] and phrase1[i][1] == phrase2[j][1]:
                has_overlap = True
                break
        if has_overlap:
            break
    return has_overlap

def load_data(file_name):
    return pickle.load(open(file_name, 'rb'))



def is_number(s):
    '''
    چک می‌کند آیا متن داده شده یک عدد است یا خیر
    :param s: متن ورودی
    :return: صحیح اگر ورودی یک عدد است و در غیر این صورت غلط
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """

    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def remove_padding(data, seq_len):
    num_seq = len(seq_len)
    nonpadded_data = []
    for i in range(num_seq):
        end = seq_len[i]
        nonpadded_data.append(data[i][:end])
    return nonpadded_data

def get_all_files(path, format):
    files = []
    for file_name in listdir(path):
        file_path = join(path, file_name)
        if isfile(file_path) and file_path.endswith(format):
            files.append(file_path)
    return files

def convert_pairs_to_clusters(pairs):
    clusters = []
    for pair in pairs:
        found = False
        for i in range(len(clusters)):
            cluster = clusters[i]
            if pair[0] in cluster or pair[1] in cluster:
                clusters[i].update(pair)
                found = True

        if not found:
            clusters.append(set(pair))
    return clusters

