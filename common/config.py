path_files_base_directory = "files/"
path_data = path_files_base_directory + "data/"
path_data_train = path_data + "train/"


path_vocabulary = path_files_base_directory + "words.vocab"

path_full_word_embedding = path_files_base_directory + "we.vec"

path_trimmed_word_embedding = path_files_base_directory + "we.trimmed"

word_embedding_dimension = 300
char_embedding_dimension = 20

vocab_min_count = 5 # کمترین تعداد دفعات تکرار یک کلمه برای اینکه آن کلمه در واژگان در نظر گرفته شود
word_max_size = 30 # بیشترین تعداد حروف یک کلمه
phrase_max_size = 10 # بیشترین تعداد کلمات یک عبارت

phrase_max_gap = 100 # تعداد کلماتی که می‌تواند بین دو عبارتی که هم‌مرجع هم هستند فاصله بیفتد

max_epoch_number = 50

path_tensorboard = "tensorboard/"

lstm_hidden_size = 128