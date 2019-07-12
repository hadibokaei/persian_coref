path_files_base_directory = "files/"
path_data = path_files_base_directory + "data/"
path_data_train = path_data + "train/"


path_vocabulary = path_files_base_directory + "words.vocab"


vocab_min_count = 5 # کمترین تعداد دفعات تکرار یک کلمه برای اینکه آن کلمه در واژگان در نظر گرفته شود
word_max_size = 30 # بیشترین تعداد حروف یک کلمه
phrase_max_size = 10 # بیشترین تعداد کلمات یک عبارت