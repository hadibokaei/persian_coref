from common import config
from os import listdir
from os.path import isdir, isfile, join
from common.vocabulary import Vocabulary

from common.utility import *


data_files_path = []

data_files_path += get_all_files(config.path_data_train, '.raw')
for file_name in listdir(config.path_data_train):
    file_path = join(config.path_data_train, file_name)
    if isdir(file_path):
        data_files_path += get_all_files(file_path, '.raw')

# Step1:
# ساخت واژگان
# data_files_path = data_files_path[:10]
vocab = Vocabulary()
if vocab.check_if_vocab_exists(config.path_vocabulary):
    answer = input("vocab exists. load(press 'l') or build(press anything other than 'l') again:")
    if answer == "l":
        vocab.load_vocab(config.path_vocabulary)
    else:
        vocab.build_vocab(data_files_path)
        vocab.dump_vocab(config.path_vocabulary)
else:
    vocab.build_vocab(data_files_path)
    vocab.dump_vocab(config.path_vocabulary)

# Step2:
# تبدیل کلمات داخت متون به اندیس واژگان: فایل‌ها را یکی یکی خوانده و پس از تبدیل به اندیس در یک فایل جدید با پسوند «ایند» می‌نویسد
def extract_span_id(text):
    text = text.replace('*','')
    text = text.replace('(','')
    text = text.replace(')','')
    return int(text)

counter = 0
for file in data_files_path:
    counter += 1
    logger.info("{}/{} start to convert the file numpy arrays: {}".format(counter, len(data_files_path), file))
    convert_to_numpy_array(file, file + ".pcl", vocab)




# '''
# نکته مهم:
# ۲ نوع شناسه برای تمام عبارات وجود دارد:
# ۱. شناسه ترتیبی (seq)
# همان شناسه‌ای که در داخل فایل‌های برچسب‌گذاری شده به عبارت تخصیص داده شده است.
# ۲. شناسه اصلی (glob)
# شناسه‌ای که با در نظر گرفتن تمام عبارات ممکن ساخته می‌شود. مثلا یک متن با ۵ کلمه، ۱۵ عبارت مختلف می‌تواند داشته باشد و عبارات از ۰ الی ۱۴ شماره‌گذاری می‌شوند
# '''
# file_counter = 0
# for file in data_files_path:
#     out_file_ind = file + ".ind" # فایلی است که در آن ۳ ستون وجود دارد: شناسه کلمه، شناسه ترتیبی عبارت و شناسه ترتیبی عبارتی که عبارت فعلی به آن اشاره می‌کند
#     out_file_clusters = file + ".clstrs" # فایلی است که در آن خوشه‌های حاوی عبارات با شناسه اصلی آن‌ها ذخیره شده است. هر خوشه در یک سطر نوشته شده است
#     out_file_clusters_info = file + ".clstrs.info" # اطلاعات هر خوشه حاوی ۴ ستون اصلی و تعداد دلخواه ستون دیگر: شناسه اصلی، شناسه ترتیبی، ترتیب کلمه شروع، ترتیب کلمه پایانی و بقیه ستون‌ها شناسه کلمات
#     out_file_info = file + ".info" # اطلاعات مربوط به فایل مثل تعداد کلمات، تعداد کلاسترها و ...
#
#     file_counter += 1
#
#     logger.info("{}/{} create processed files for: {}".format(file_counter, len(data_files_path), file))
#
#     cur_span_id = 0
#     cur_ref_id = 0
#     out_lines = []
#     span_ref = dict() #کلید این دیکشنری آی‌دی عبارت و مقدار آن عددی است که نشان‌دهنده ارجاع این عبارت است
#     span_word_ids = dict() #کلید این دیکشنری آی‌دی عبارت و مقدار آن یک لیست است که به ترتیب آی‌دی کلمات داخل عبارت در آن مشخص شده‌اند
#     clusters_seq_ids = [] # لیستی است که هر آیتم در آن یک خوشه‌ای از تمام عباراتی است که به یک موجودیت ارجاع دارند. شناسه عبارات شناسه ترتیبی آن‌ها است
#     clusters_glob_ids = [] # لیستی است که هر آیتم در آن یک خوشه‌ای از تمام عباراتی است که به یک موجودیت ارجاع دارند. شناسه عبارات شناسه اصلی آن‌ها است
#     span_start_end_word_order = dict() # کلید این دیکشنری شناسه ترتیبی عبارات و مقدار آن یک دوتایی است که آیتم اول آن ردیف کلمه شروع‌کننده عبارت و آیتم دوم ردیف کلمه انتهایی عبارت را مشخص می‌کند
#     span_seq2glob_map = dict() # شناسه ترتیبی را به شناسه اصلی تبدیل می‌کند
#     span_glob2seq_map = dict() # شناسه اصلی را به شناسه ترتیبی تبدیل می‌کند
#
#     word_counter = 0
#     span_start = 0
#     span_end = 0
#     with open(file, 'r') as f:
#         for line in f.readlines():
#             splitted_line = line.split()
#             if(len(splitted_line) == 0):
#                 continue
#             word = splitted_line[0]
#             word_id = vocab.get_word_id(word, config.vocab_min_count)
#
#             if(splitted_line[2].endswith('(*')):
#                 span_id = extract_span_id(splitted_line[2])
#                 span_start = word_counter
#                 cur_span_id = span_id
#             elif(splitted_line[2] == '*'):
#                 span_id = cur_span_id
#             elif(splitted_line[2].endswith('*)')):
#                 span_id = cur_span_id
#                 span_end = word_counter
#                 span_start_end_word_order[span_id] = (span_start, span_end)
#             elif (splitted_line[2] == '-'):
#                 span_id = 0
#             else:
#                 span_start = word_counter
#                 span_end = word_counter
#                 span_id = int(splitted_line[2])
#                 span_start_end_word_order[span_id] = (span_start, span_end)
#
#             if(splitted_line[3].endswith('(*')):
#                 ref_id = extract_span_id(splitted_line[3])
#                 cur_ref_id = ref_id
#             elif(splitted_line[3] == '*'):
#                 ref_id = cur_ref_id
#             elif(splitted_line[3].endswith('*)')):
#                 ref_id = cur_ref_id
#             elif (splitted_line[3] == '-'):
#                 ref_id = 0
#             else:
#                 ref_id = int(splitted_line[3])
#
#             if ref_id == span_id:
#                 ref_id = 0
#
#             if span_id > 0:
#                 if span_id not in span_word_ids.keys():
#                     span_word_ids[span_id] = []
#                 span_word_ids[span_id].append(word_id)
#
#             span_ref[span_id] = ref_id
#
#             if ref_id == 0 and span_id == 0:
#                 pass
#             elif ref_id == 0:
#                 found = False
#                 for i in range(len(clusters_seq_ids)):
#                     if span_id in clusters_seq_ids[i]:
#                         found = True
#                 if not found:
#                     clusters_seq_ids.append([span_id])
#             else:
#                 found = False
#                 for i in range(len(clusters_seq_ids)):
#                     if ref_id in clusters_seq_ids[i]:
#                         if span_id not in clusters_seq_ids[i]:
#                             clusters_seq_ids[i].append(span_id)
#                         found = True
#                 if not found:
#                     clusters_seq_ids.append([ref_id, span_id])
#
#             out_lines.append("{}\t{}\t{}\n".format(word_id, span_id, ref_id))
#
#             word_counter += 1
#
#     for k in span_start_end_word_order.keys():
#         old_id = k
#         start_word_order = span_start_end_word_order[k][0]
#         end_word_order = span_start_end_word_order[k][1]
#         new_id = int((2*start_word_order*word_counter - start_word_order^2 - start_word_order +2*end_word_order)/2)
#         span_glob2seq_map[new_id] = old_id
#         span_seq2glob_map[old_id] = new_id
#
#     all_span_glob_ids = []
#     for l in clusters_seq_ids:
#         l_glob = sorted([span_seq2glob_map[i] for i in l])
#         clusters_glob_ids.append(l_glob)
#         all_span_glob_ids += l_glob
#     all_span_glob_ids = sorted(all_span_glob_ids)
#
#     with open(out_file_ind, 'w') as f:
#         f.writelines(out_lines)
#
#     with open(out_file_clusters, 'w') as f:
#         for clstr in clusters_glob_ids:
#             f.write('{}\n'.format(' '.join(str(x) for x in clstr)))
#
#     with open(out_file_clusters_info, 'w') as f:
#         for span_glob_id in all_span_glob_ids:
#             span_seq_id = span_glob2seq_map[span_glob_id]
#             span_start_word_order, span_end_word_order = span_start_end_word_order[span_seq_id]
#             span_words = span_word_ids[span_seq_id]
#             f.write('{} {} {} {} {}\n'.format(span_glob_id, span_seq_id, span_start_word_order, span_end_word_order
#                                               , ' '.join(str(x) for x in span_words)))
#
#     with open(out_file_info, 'w') as f:
#         f.write("num words:\t{}\n".format(word_counter))
#         f.write("num clusters:\t{}\n".format(len(clusters_glob_ids)))
#         f.write("num spans:\t{}\n".format(len(all_span_glob_ids)))
#
