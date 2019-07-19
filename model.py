import tensorflow as tf
from  common.utility import pad_sequences, logger
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import product

class CorefModel(object):

    def __init__(self, word_vocab_size, char_vocab_size, word_embedding_dimension, char_embedding_dimension, max_word_length, max_phrase_length
                 , conv_filter_num, conv_filter_size
                 , lstm_unit_size, dir_tensoboard_log):
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embedding_dimension = word_embedding_dimension
        self.char_embedding_dimension = char_embedding_dimension
        self.max_word_length = max_word_length
        self.conv_filter_num = conv_filter_num
        self.conv_filter_size = conv_filter_size
        self.lstm_unit_size = lstm_unit_size
        self.max_phrase_length = max_phrase_length
        self.dir_tensoboard_log = dir_tensoboard_log

    def build_graph(self):
        self.add_placeholders()
        self.add_word_representation()
        self.add_lstm()
        self.add_fcn_phrase()
        self.add_phrase_loss_train()
        self.add_pair_processing()
        self.add_fcn_pair()
        self.add_pair_loss_train()
        self.add_final_train()
        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.summary.FileWriter(self.dir_tensoboard_log, graph=tf.get_default_graph())

    def add_placeholders(self):
        self.word_ids           = tf.placeholder(tf.int32, shape=[None, None], name="word_ids") #shape=[# of sentences in doc, max # of words in sentences]
        self.word_embedding     = tf.placeholder(tf.float32, shape=[self.word_vocab_size, self.word_embedding_dimension], name="word_embedding") #shape=[vocab size, embedding dimension]
        self.sentence_length    = tf.placeholder(tf.int32, shape=[None], name="sentence_length") #shape=[# of sentences in doc]
        self.char_ids           = tf.placeholder(tf.int32, shape=[None, None, self.max_word_length]) #shape=[# of sentences in doc, max # of words in sentences, max # of characters in a word]
        self.word_length        = tf.placeholder(tf.int32, shape=[None, None], name="word_length") #shape=[# of sentences in doc, max # of words in sentences]
        self.phrase_indices     = tf.placeholder(tf.int32, shape=[None, self.max_phrase_length, 2]) #shape=[# of candidate phrases in doc, max phrase length in words, 2]
        self.gold_phrases       = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate phrases in doc]
        self.phrase_length      = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate phrases in doc]
        self.phrase_weights     = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate phrases in doc]
        self.pair_rep_indices   = tf.placeholder(tf.int64, shape=[None, 2, 1]) #shape=[# of candidate pairs in doc, 2, 1]
        self.pair_gold          = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate pairs in doc]
        self.pruned_cand_pair   = tf.placeholder(tf.int32, shape=[]) #scalar
        self.pair_weights       = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate pairs]


    def add_word_representation(self):
        embedded_words = tf.nn.embedding_lookup(self.word_embedding, self.word_ids, name="embedded_words") #shape=[# of sentences in doc, max # of words in sentences, word embedding dimension]
        char_embedding = tf.get_variable(dtype=tf.float32, shape=[self.char_vocab_size, self.char_embedding_dimension], name="char_embeddings")
        embedded_chars = tf.nn.embedding_lookup(char_embedding, self.char_ids, name='embedded_chars') #shape=[# of sentences in doc, max # of words in sentences, max number of characters in a word, char embedding dimension]
        embedded_chars_shape = tf.shape(embedded_chars)

        embedded_chars = tf.reshape(embedded_chars, shape=[-1, self.max_word_length, self.char_embedding_dimension]) #shape=[# of sentences * max num of words in each sentence, max word length, char embedding dimension]
        embedded_chars = tf.expand_dims(embedded_chars, -1) #shape=[# of sentences * max num of words in each sentence, max word length, char embedding dimension, 1]


        pooled_output = []

        for fs in self.conv_filter_size:
            conv = tf.keras.layers.Conv2D(filters=self.conv_filter_num, kernel_size=fs, padding='same', data_format='channels_last')(embedded_chars) #shape=[# of sentences * max num of words in each sentence, max word length, char embedding dimension, filter size]
            pool = tf.keras.layers.MaxPool2D(pool_size=[self.max_word_length, 1])(conv) # shape=[# of sentences * max num of words in each sentence, 1, char embedding dimension, filter size]
            pool = tf.reshape(pool, shape=[embedded_chars_shape[0], embedded_chars_shape[1], self.char_embedding_dimension*self.conv_filter_num] ) # shape=[# of sentences, max num of words in each sentence, char embedding dimension * # of filters]
            pooled_output.append(pool)

        concat_pooled = tf.concat(pooled_output, 2) # shape = [# of sentences, max num of words in each sentence, char embedding dimension * # of filters * # of different filter sizes]

        self.word_representation = tf.concat([concat_pooled, embedded_words], 2) # shape = [# of sentences, max num of words in each sentence, char embedding dimension * # of filters * # of different filter sizes + word embedding dimension]

    def add_lstm(self):
        with tf.variable_scope('sentence_bilstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_unit_size)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_unit_size)
            (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_representation,
                sequence_length=self.sentence_length,
                dtype=tf.float32)
            lstm_output_tmp = tf.concat([outputs_fw, outputs_bw], axis=2) # shape = [# of sentences, max num of words in each sentence, 2 * lstm hidden size]

        # cell_fw = tf.keras.layers.LSTM(units=self.lstm_unit_size, activation='relu', return_sequences=True)
        # cell_bw = tf.keras.layers.LSTM(units=self.lstm_unit_size, activation='relu', return_sequences=True, go_backwards=True)
        # lstm_output_tmp = tf.keras.layers.Bidirectional(layer=cell_fw, backward_layer=cell_bw, merge_mode='concat')(self.word_representation) # shape = [# of sentences, max num of words in each sentence, 2 * lstm hidden size]

        self.lstm_output = tf.concat([tf.expand_dims(tf.zeros_like(lstm_output_tmp[0]),0),lstm_output_tmp], axis = 0) # shape = [# of sentences + 1, max num of words in each sentence, 2 * lstm hidden size]

        self.candidate_phrases = tf.gather_nd(self.lstm_output, self.phrase_indices) # shape = [# of candidate phrases, max phrase length, 2 * lstm hidden size

        with tf.variable_scope('phrase_bilstm'):
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_unit_size, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.lstm_unit_size, state_is_tuple=True)
            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.candidate_phrases,
                sequence_length=self.phrase_length,
                dtype=tf.float32)
            self.phrase_rep = tf.concat([output_fw, output_bw], axis=-1) # shape = [# of candidate phrases, 2 * lstm hidden size]

    def add_fcn_phrase(self):
        dense_output = tf.keras.layers.Dense(self.lstm_unit_size,activation='relu')(self.phrase_rep) # shape = [# of candidate phrases, lstm hidden size]
        self.candidate_phrase_probability = tf.squeeze(tf.keras.layers.Dense(1, activation='sigmoid')(dense_output)) # shape = [# of candidate phrases]

    def add_pair_processing(self):
        pair_rep = tf.reshape(tf.gather_nd(self.phrase_rep, self.pair_rep_indices)
                              , shape=[tf.shape(self.pair_rep_indices)[0], 4*self.lstm_unit_size]) # shape = [# of candidate pairs, 4 * lstm hidden size]
        pair_score = tf.gather_nd(self.candidate_phrase_probability, self.pair_rep_indices) # shape = [# of candidate pairs, 2]
        pair_min_score = tf.reduce_min(pair_score, axis=1) # shape = [# of candidate pairs]
        pair_candidate_indices = tf.expand_dims(tf.math.top_k(pair_min_score, k=self.pruned_cand_pair).indices, 1)

        pair_candidate_indices = tf.cast(pair_candidate_indices, tf.int64)

        self.pair_pruned_gold = tf.gather_nd(self.pair_gold, pair_candidate_indices) #shape=[# of pruned candidate pairs in doc]
        self.pair_min_pruned_score = tf.gather_nd(pair_min_score, pair_candidate_indices) # shape = [# of pruned candidate pairs]
        self.pair_pruned_weights = tf.gather_nd(self.pair_weights, pair_candidate_indices) # shape = [# of pruned candidate pairs]
        self.pair_pruned_rep = tf.gather_nd(pair_rep, pair_candidate_indices) # shape = [# of pruned candidate pairs, 4 * lstm hidden size]

        self.pair_pruned_rep = tf.reshape(self.pair_pruned_rep, shape=[-1, 4 *self.lstm_unit_size])

    def add_fcn_pair(self):
        dense_output = tf.keras.layers.Dense(self.lstm_unit_size,activation='relu')(self.pair_pruned_rep) # shape = [# of pruned candidate pairs, lstm hidden size]
        self.candidate_pair_probability = tf.squeeze(tf.keras.layers.Dense(1, activation='sigmoid')(dense_output)) # shape = [# of pruned candidate pairs]

    def add_phrase_loss_train(self):

        gold = tf.expand_dims(tf.to_float(self.gold_phrases),1)
        gold_2d = tf.concat([gold,1-gold],1)

        pred = tf.expand_dims(self.candidate_phrase_probability,1)
        pred_2d = tf.concat([pred,1-pred],1)

        w = tf.expand_dims(self.phrase_weights,1)

        self.phrase_identification_loss = tf.losses.sigmoid_cross_entropy(gold_2d, pred_2d, w)

        self.phrase_identification_train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.phrase_identification_loss)

    def add_pair_loss_train(self):

        gold = tf.expand_dims(tf.to_float(self.pair_pruned_gold),1)
        gold_2d = tf.concat([gold,1-gold],1)

        pred = tf.expand_dims(self.candidate_pair_probability,1)
        pred_2d = tf.concat([pred,1-pred],1)

        w = tf.expand_dims(self.pair_pruned_weights,1)

        self.pair_identification_loss = tf.losses.sigmoid_cross_entropy(gold_2d, pred_2d, w)

        self.pair_identification_train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.pair_identification_loss)

    def add_final_train(self):
        self.final_loss = self.phrase_identification_loss + 5* self.pair_identification_loss
        self.final_train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.final_loss)

    def train_phrase_identification(self, word_embedding, all_docs_word_ids, all_docs_char_ids, all_docs_phrase_indices
                                    , all_docs_gold_phrases, all_docs_phrase_length, epoch_start, max_epoch_number):
        for epoch in range(epoch_start, max_epoch_number):
            for batch_number in range(len(all_docs_word_ids)):
                current_word_ids = all_docs_word_ids[batch_number]

                current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

                current_char_ids = all_docs_char_ids[batch_number]
                current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

                current_gold_phrase = all_docs_gold_phrases[batch_number]
                weight = len(current_gold_phrase)/(4*np.sum(current_gold_phrase))

                current_weight = current_gold_phrase*weight + 1

                feed_dict = {
                    self.word_ids: current_word_ids,
                    self.word_embedding: word_embedding,
                    self.sentence_length: current_sentence_length,
                    self.char_ids: current_char_ids,
                    self.word_length: current_word_length,
                    self.phrase_indices: all_docs_phrase_indices[batch_number],
                    self.gold_phrases: current_gold_phrase,
                    self.phrase_length: all_docs_phrase_length[batch_number],
                    self.phrase_weights: current_weight
                }
                [_, loss, pred] = self.sess.run([self.phrase_identification_train, self.phrase_identification_loss, self.candidate_phrase_probability], feed_dict)

                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                gold = all_docs_gold_phrases[batch_number]

                precision = precision_score(gold, pred) * 100
                recall = recall_score(gold, pred) * 100
                f1_measure = f1_score(gold, pred) * 100
                logger.info("epoch:{:3d} batch:{:4d} loss:{:5.3f} precision:{:5.2f} recall:{:5.2f} f1:{:5.2f}"
                            .format(epoch, batch_number, loss, precision, recall, f1_measure))

                # a = pred[all_docs_gold_phrases[batch_number]==1]
                # print(a[:5])
                # b = pred[all_docs_gold_phrases[batch_number]==0]
                # print(b[:5])

    def train(self, word_embedding, all_docs_word_ids, all_docs_char_ids, all_docs_phrase_indices
              , all_docs_gold_phrases, all_docs_phrase_length
              , all_docs_pair_indices, all_docs_pair_golds
              , epoch_start, max_epoch_number):
        for epoch in range(epoch_start, max_epoch_number):
            for batch_number in range(len(all_docs_word_ids)):
                current_word_ids = all_docs_word_ids[batch_number]

                current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

                current_char_ids = all_docs_char_ids[batch_number]
                current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

                current_gold_phrase = all_docs_gold_phrases[batch_number]

                phrase_weight = len(current_gold_phrase)/(4*np.sum(current_gold_phrase))
                current_phrase_weight = current_gold_phrase*phrase_weight + 1

                current_gold_pair = all_docs_pair_golds[batch_number]

                pair_weight = len(current_gold_pair)/(150*np.sum(current_gold_pair)) #30: Too high 15: Too low
                # pair_weight = 1
                current_pair_weight = current_gold_pair*pair_weight + 1

                pruned_cand_pair = int(len(current_gold_pair)/100)

                # logger.info("sentences:{} candidate phrases:{} gold phrases:{} candidate pairs:{} gold pairs:{} pruned pair:{}"
                #             .format(len(current_word_ids), len(current_gold_phrase), np.sum(current_gold_phrase), np.shape(all_docs_pair_indices[batch_number]),
                #                     np.sum(current_gold_pair), pruned_cand_pair))


                feed_dict = {
                    self.word_ids: current_word_ids,
                    self.word_embedding: word_embedding,
                    self.sentence_length: current_sentence_length,
                    self.char_ids: current_char_ids,
                    self.word_length: current_word_length,
                    self.phrase_indices: all_docs_phrase_indices[batch_number],
                    self.gold_phrases: current_gold_phrase,
                    self.phrase_length: all_docs_phrase_length[batch_number],
                    self.phrase_weights: current_phrase_weight,
                    self.pair_gold: current_gold_pair,
                    self.pair_rep_indices: all_docs_pair_indices[batch_number],
                    self.pair_weights: current_pair_weight,
                    self.pruned_cand_pair: pruned_cand_pair
                }
                [_, loss, pred, gold] = self.sess.run([self.final_train, self.final_loss
                                                          , self.candidate_pair_probability, self.pair_pruned_gold], feed_dict)

                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                precision = precision_score(gold, pred) * 100
                recall = recall_score(gold, pred) * 100
                f1_measure = f1_score(gold, pred) * 100
                logger.info("epoch:{:3d} batch:{:4d} loss:{:5.3f} precision:{:5.2f} recall:{:5.2f} f1:{:5.2f}"
                            .format(epoch, batch_number, loss, precision, recall, f1_measure))

                print("orig gold:{}/{} pruned gold:{}/{} pred:{}/{}"
                      .format(np.sum(current_gold_pair), len(current_gold_pair)
                              , np.sum(gold), len(gold)
                              , np.sum(pred), len(pred)))

                # a = pred[gold==1]
                # print(a[:5])
                # b = pred[gold==0]
                # print(b[:5])


# phrase_rep = tf.constant([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
# pair_rep_indices = tf.constant([[[0],[1]],[[0],[2]],[[0],[3]],[[1],[2]],[[1],[3]],[[2],[3]]])
# pair_gold = tf.constant([0,0,1,0,0,1])
# phrase_scores = tf.constant([3,1,4,5])
#
# pair_rep = tf.reshape(tf.gather_nd(phrase_rep, pair_rep_indices), shape=[6,6])
# pair_score = tf.gather_nd(phrase_scores, pair_rep_indices)
# pair_min_score = tf.reduce_min(pair_score,axis = 1)
#
# pair_candidate_indices = tf.expand_dims(tf.math.top_k(pair_min_score, k=4).indices, 1)
#
#
#
# pair_pruned_rep = tf.gather_nd(pair_rep, pair_candidate_indices)
# pair_pruned_gold = tf.gather_nd(pair_gold, pair_candidate_indices)
# pair_min_pruned_score = tf.gather_nd(pair_min_score, pair_candidate_indices)
#
# pair_rep.eval()
# pair_score.eval()
# pair_min_score.eval()
# pair_candidate_indices.eval()
#
# pair_pruned_rep.eval()
# pair_pruned_gold.eval()
# pair_min_pruned_score.eval()





