from model import CorefModel
import tensorflow as tf

print(tf.__version__)
model = CorefModel(word_vocab_size=100, char_vocab_size=10, word_embedding_dimension=16
                   , char_embedding_dimension=4, max_word_length=5, conv_filter_num=2, conv_filter_size=[2,3,4,5], lstm_unit_size=9)
model.build_graph()