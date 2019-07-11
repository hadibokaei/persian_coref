from common.utility import logger
import pickle
from os.path import isfile

class Vocabulary(object):

    def __init__(self):
        self.id2word = dict()
        self.word2id = dict()
        self.vocab_size = 0
        self.id2count = dict()
        self.OOV_string = "<oov>"
        self.OOV_id = 0
        self.chars = set()
        self.id2char = dict()
        self.char2id = dict()
        self.char_size = 0

    def build_vocab(self, files):
        '''
        ساخت واژگان از متون ذخیره شده در فایل‌های ورودی. این تابع باید برای تسک‌ها و فرمت‌های مختلف ورودی بازنویسی شود
        :param files: لیستی از فایل‌های حاوی متن با فرمت مشخص
        :return:
        '''
        logger.info("start to build vocabulray:")
        counter = 0
        for file in files:
            with open(file, 'r') as f:
                counter += 1
                logger.info("process file: {}/{}\r".format(counter, len(files)))
                for line in f.readlines():
                    splitted_line = line.split()
                    if len(splitted_line) == 0:
                        continue
                    word = splitted_line[0]
                    if word not in self.word2id.keys():
                        self.vocab_size += 1
                        id = self.vocab_size
                        self.id2word[id] = word
                        self.word2id[word] = id
                        self.id2count[id] = 1
                        self.chars.update(word)
                        for letter in word:
                            if letter not in self.char2id.keys():
                                self.char_size += 1
                                char_id = self.char_size
                                self.id2char[char_id] = letter
                                self.char2id[letter] = char_id
                    else:
                        id = self.word2id[word]
                        self.id2count[id] += 1

    def get_file_paths(self, file_path):
        '''
        یک آدرس فایل (با عنوان آدرس مادر) در متدهای مختلف این کلاس گرفته می‌شود. اما این آدرس فایل به دو جا اشاره می‌کند: فایلی که کل آبجکت به صورت پیکل ذخیره شده و فایلی که انسان می‌تواند آن را بخواند و شناسه، کلمه و تعداد تکرار کلمات را ببیند
        :param file_path: آدرس فایل اصلی که باید به ۲ فایل تبدیل شود
        :return:
        '''
        return file_path + ".pckl", file_path +".txt", file_path + ".chars.txt"

    def dump_vocab(self, file_path):
        '''
        واژگان ساخته شده را به دو فرمت پیکل و قابل خواندن ذخیره می‌کند
        :param file_path: آدرس مادر
        :return:
        '''
        logger.info("dump vocabulary in {}".format(file_path))
        [pickle_file_path, human_readable_file_path, char_file_path] = self.get_file_paths(file_path)

        with open(pickle_file_path, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)

        with open(human_readable_file_path, 'w') as f:
            for id in self.id2word.keys():
                f.write('{}\t{}\t{}\r\n'.format(id, self.id2word[id], self.id2count[id]))

        with open(char_file_path, 'w') as f:
            for char in sorted(self.chars):
                f.write('{}\r\n'.format(char))

    def load_vocab(self, file_path):
        '''
        واژگانی که قبلا ذخیره شده لود می‌شود
        :param file_path: آدرس مادر
        :return:
        '''
        logger.info("load vocabulary from {}".format(file_path))
        [pickle_file_path, _, _] = self.get_file_paths(file_path)

        f = open(pickle_file_path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def check_if_vocab_exists(self, file_path):
        '''
        بررسی می‌کند آیا واژگانی قبلا در محل مشخص‌شده نوشته شده است یا خیر
        :param file_path: آدرس مادر
        :return: صحیح اگر وجود داشته باشد و غلط اگر وجود نداشته باشد
        '''
        logger.info("check if vocabulary exists in {}".format(file_path))
        [pickle_file_path, human_readable_file_path, _] = self.get_file_paths(file_path)

        if isfile(pickle_file_path) and isfile(human_readable_file_path):
            return True
        else:
            return False

    def get_word_id(self, word, thresh):
        '''
        شناسه یک کلمه را برمی‌گرداند
        :param word: کامه مورد نظر
        :param thresh: تعداد دفعات تکرار آستانه. اگر از این میزان کمتر باشد شناسه کلمه خارج از واژگان را بر می‌گرداند
        :return:
        '''
        id = self.word2id[word]
        count = self.id2count[id]
        if(count > thresh):
            return id
        else:
            return self.OOV_id

    def get_word(self, id):
        '''
        شناسه مربوط به یک کلمه را برمی‌گرداند
        :param id: شناسه کله
        :return:
        '''
        if id==self.OOV_id:
            return self.OOV_string
        else:
            return self.id2word[id]

    def get_char_id(self, char):
        id = self.char2id[char]
        return id
