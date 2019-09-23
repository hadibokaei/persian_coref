import tensorflow as tf
from  common.utility import pad_sequences, logger, load_data
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import product
import random

class CorefModel(object):

    def __init__(self, word_vocab_size, char_vocab_size, word_embedding_dimension, char_embedding_dimension, max_word_length, max_phrase_length
                 , conv_filter_num, conv_filter_size
                 , lstm_unit_size, dir_tensoboard_log, dir_checkpoint):
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
        self.dir_checkpoint = dir_checkpoint

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
        self.merged = tf.summary.merge_all()

    def initialize_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=3)
        self.train_writer = tf.summary.FileWriter(self.dir_tensoboard_log + "train", graph=tf.get_default_graph())
        self.validation_writer = tf.summary.FileWriter(self.dir_tensoboard_log + "validation", graph=tf.get_default_graph())


    def add_placeholders(self):
        self.word_ids           = tf.placeholder(tf.int32, shape=[None, None], name="word_ids") #shape=[# of sentences in doc, max # of words in sentences]
        self.word_embedding     = tf.placeholder(tf.float32, shape=[self.word_vocab_size, self.word_embedding_dimension], name="word_embedding") #shape=[vocab size, embedding dimension]
        self.sentence_length    = tf.placeholder(tf.int32, shape=[None], name="sentence_length") #shape=[# of sentences in doc]
        self.char_ids           = tf.placeholder(tf.int32, shape=[None, None, self.max_word_length]) #shape=[# of sentences in doc, max # of words in sentences, max # of characters in a word]
        self.word_length        = tf.placeholder(tf.int32, shape=[None, None], name="word_length") #shape=[# of sentences in doc, max # of words in sentences]
        self.phrase_indices     = tf.placeholder(tf.int32, shape=[None, self.max_phrase_length, 2]) #shape=[# of candidate phrases in doc, max phrase length in words, 2]
        self.gold_phrases       = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate phrases in doc]
        self.phrase_length      = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate phrases in doc]
        self.pair_rep_indices   = tf.placeholder(tf.int64, shape=[None, 2, 1]) #shape=[# of candidate pairs in doc, 2, 1]
        self.pair_gold          = tf.placeholder(tf.int32, shape=[None]) #shape=[# of candidate pairs in doc]
        self.dropout_rate       = tf.placeholder(tf.float32, shape=[]) #scalar
        self.learning_rate      = tf.placeholder(tf.float32, shape=[]) #scalar


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

        self.candidate_phrases = tf.gather_nd(self.lstm_output, self.phrase_indices) # shape = [# of candidate phrases, max phrase length, 2 * lstm hidden size]

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
            tf.summary.histogram("phrase representation", self.phrase_rep)

    def add_fcn_phrase(self):

        dropped_rep = tf.keras.layers.Dropout(rate = self.dropout_rate)(self.phrase_rep)
        dense_output = tf.keras.layers.Dense(self.lstm_unit_size,activation='elu')(dropped_rep) # shape = [# of candidate phrases, lstm hidden size]
        tf.summary.histogram("output layer", dense_output)
        dropped_dense_output = tf.keras.layers.Dropout(rate = self.dropout_rate)(dense_output)
        self.candidate_phrase_logit = tf.squeeze(tf.keras.layers.Dense(1, activation='elu')(dropped_dense_output)) # shape = [# of candidate phrases]
        self.candidate_phrase_probability = tf.math.sigmoid(self.candidate_phrase_logit)

        pred = tf.to_int32(self.candidate_phrase_probability > 0.5)

        with tf.name_scope('metrics'):
            accuracy, accuracy_op = tf.metrics.accuracy(labels=self.gold_phrases, predictions=pred)
            precision, precision_op = tf.metrics.precision(labels=self.gold_phrases, predictions=pred)
            recall, recall_op = tf.metrics.recall(labels=self.gold_phrases, predictions=pred)


        tf.summary.scalar("accuracy", accuracy_op)
        tf.summary.scalar("precision", precision_op)
        tf.summary.scalar("recall", recall_op)


    def add_pair_processing(self):
        self.pair_rep = tf.reshape(tf.gather_nd(self.phrase_rep, self.pair_rep_indices)
                              , shape=[tf.shape(self.pair_rep_indices)[0], 4*self.lstm_unit_size]) # shape = [# of candidate pairs, 4 * lstm hidden size]

    def add_fcn_pair(self):
        dropped_rep = tf.keras.layers.Dropout(rate = self.dropout_rate)(self.pair_rep)
        dense_output = tf.keras.layers.Dense(self.lstm_unit_size,activation='elu')(dropped_rep) # shape = [# of pruned candidate pairs, lstm hidden size]
        # tf.summary.histogram("pair output layer", dense_output)
        dropped_dense_output = tf.keras.layers.Dropout(rate = self.dropout_rate)(dense_output)
        self.candidate_pair_logit = tf.squeeze(tf.keras.layers.Dense(1, activation='elu')(dropped_dense_output)) # shape = [# of pruned candidate pairs]
        self.candidate_pair_probability = tf.math.sigmoid(self.candidate_pair_logit)
        pred = tf.to_int32(self.candidate_pair_probability > 0.5)

        with tf.name_scope('metrics'):
            accuracy, accuracy_op = tf.metrics.accuracy(labels=self.pair_gold, predictions=pred)
            precision, precision_op = tf.metrics.precision(labels=self.pair_gold, predictions=pred)
            recall, recall_op = tf.metrics.recall(labels=self.pair_gold, predictions=pred)


        # tf.summary.scalar("pair accuracy", accuracy_op)
        # tf.summary.scalar("pair precision", precision_op)
        # tf.summary.scalar("pair recall", recall_op)

    def add_phrase_loss_train(self):

        gold = tf.expand_dims(tf.to_float(self.gold_phrases),1)
        gold_2d = tf.concat([gold,1-gold],1)

        pred = tf.expand_dims(self.candidate_phrase_logit, 1)
        pred_2d = tf.concat([pred,1-pred],1)

        self.phrase_identification_loss = tf.losses.sigmoid_cross_entropy(gold_2d, pred_2d)
        tf.summary.scalar("phrase loss", self.phrase_identification_loss)

        self.phrase_identification_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.phrase_identification_loss)

    def add_pair_loss_train(self):

        # loss_per_point = -(tf.multiply(tf.math.log(self.candidate_pair_probability), tf.cast(self.pair_gold, tf.float32))
        #                    +tf.multiply(tf.math.log(1-self.candidate_pair_probability), 1-tf.cast(self.pair_gold, tf.float32)))

        loss_per_point = tf.multiply(self.candidate_pair_probability, tf.cast(self.pair_gold, tf.float32))

        self.pair_identification_loss = tf.reduce_sum(loss_per_point)




        # gold = tf.expand_dims(tf.to_float(self.pair_gold),1)
        # gold_2d = tf.concat([gold,1-gold],1)
        #
        # pred = tf.expand_dims(self.candidate_pair_logit, 1)
        # pred_2d = tf.concat([pred,1-pred],1)
        #
        # self.pair_identification_loss = tf.losses.sigmoid_cross_entropy(gold_2d, pred_2d, reduction=tf.losses.Reduction.SUM)

        self.pair_identification_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.pair_identification_loss)

    def add_final_train(self):
        self.final_loss = self.phrase_identification_loss + self.pair_identification_loss
        self.final_train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.final_loss)

    def train_phrase_identification(self, word_embedding, train_files_path, validation_files_path, epoch_start, max_epoch_number, learning_rate):
        global_step = 0
        for epoch in range(epoch_start, max_epoch_number):
            stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            self.sess.run(tf.variables_initializer(stream_vars_valid))
            for batch_number in range(len(train_files_path)):

                global_step += 1

                file = train_files_path[batch_number]
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, _, _] = load_data(file)
                if len(doc_word) == 0:
                    print("skip this file (zero length document): {}".format(file))
                    continue
                if np.sum(gold_phrase) == 0:
                    print("skip this file (no phrase): {}".format(file))
                    continue

                current_word_ids = doc_word
                current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

                current_char_ids = doc_char
                current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

                current_gold_phrase = gold_phrase
                num_posetive = np.sum(current_gold_phrase)

                negative_indices = np.array(random.choices(np.squeeze(np.argwhere(current_gold_phrase == 0)), k=100*num_posetive))
                posetive_indices = np.squeeze(np.argwhere(current_gold_phrase == 1))
                all_indices = np.concatenate([negative_indices, posetive_indices])
                np.random.shuffle(all_indices)
                print("+{}-{}:{}/{}".format(len(posetive_indices), len(negative_indices), len(all_indices), len(current_gold_phrase)))

                current_doc_phrase_indices = phrase_word[all_indices]
                current_doc_gold_phrases = gold_phrase[all_indices]
                current_doc_phrase_length = phrase_word_len[all_indices]

                # current_doc_phrase_indices = phrase_word
                # current_doc_gold_phrases = gold_phrase
                # current_doc_phrase_length = phrase_word_len


                feed_dict = {
                    self.word_ids: current_word_ids,
                    self.word_embedding: word_embedding,
                    self.sentence_length: current_sentence_length,
                    self.char_ids: current_char_ids,
                    self.word_length: current_word_length,
                    self.phrase_indices: current_doc_phrase_indices,
                    self.gold_phrases: current_doc_gold_phrases,
                    self.phrase_length: current_doc_phrase_length,
                    self.dropout_rate: 0.5,
                    self.learning_rate: learning_rate
                }
                try:
                    [_, loss, pred, summary] = self.sess.run([self.phrase_identification_train, self.phrase_identification_loss
                                                                 , self.candidate_phrase_probability, self.merged], feed_dict)

                    self.train_writer.add_summary(summary, global_step)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0

                    gold = current_doc_gold_phrases

                    precision = precision_score(gold, pred) * 100
                    recall = recall_score(gold, pred) * 100
                    f1_measure = f1_score(gold, pred) * 100
                    logger.info("epoch:{:3d} batch:{:4d} loss:{:5.3f} precision:{:5.2f} recall:{:5.2f} f1:{:5.2f}"
                                .format(epoch, batch_number, loss, precision, recall, f1_measure))
                except Exception as e:
                    print(e)

            save_path = self.saver.save(self.sess, "{}/coref_model".format(self.dir_checkpoint),
                                        global_step=int(epoch), write_meta_graph=False)
            logger.info("model is saved in: {}".format(save_path))

            all_precision = []
            all_recall = []
            all_f1 = []
            stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            self.sess.run(tf.variables_initializer(stream_vars_valid))
            for doc_num in range(len(validation_files_path)):


                file = validation_files_path[doc_num]
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, _, _] = load_data(file)

                if len(doc_word) == 0:
                    print("skip this file (zero length document): {}".format(file))
                    continue
                if np.sum(gold_phrase) == 0:
                    print("skip this file (no phrase): {}".format(file))
                    continue


                current_word_ids = doc_word
                current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

                current_char_ids = doc_char
                current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

                current_doc_phrase_indices = phrase_word
                current_doc_gold_phrases = gold_phrase
                current_doc_phrase_length = phrase_word_len

                feed_dict = {
                    self.word_ids: current_word_ids,
                    self.word_embedding: word_embedding,
                    self.sentence_length: current_sentence_length,
                    self.char_ids: current_char_ids,
                    self.word_length: current_word_length,
                    self.phrase_indices: current_doc_phrase_indices,
                    self.gold_phrases: current_doc_gold_phrases,
                    self.phrase_length: current_doc_phrase_length,
                    self.dropout_rate: 0.0
                }
                try:
                    [pred, summary] = self.sess.run([self.candidate_phrase_probability, self.merged], feed_dict)

                    self.validation_writer.add_summary(summary, global_step)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0

                    gold = current_doc_gold_phrases

                    precision = precision_score(gold, pred) * 100
                    all_precision.append(precision)
                    recall = recall_score(gold, pred) * 100
                    all_recall.append(recall)
                    f1_measure = f1_score(gold, pred) * 100
                    all_f1.append(f1_measure)
                except Exception as e:
                    print(e)

            avg_precision = np.average(all_precision)
            avg_recall = np.average(all_recall)
            avg_f1 = np.average(all_f1)

            logger.info("==================================>epoch:{:3d} validation metrics: precision:{:5.2f} recall:{:5.2f} f1:{:5.2f}".format(epoch, avg_precision, avg_recall, avg_f1))

    def restore_graph(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.dir_checkpoint))
        return tf.train.latest_checkpoint(self.dir_checkpoint)

    def train_pair_identification(self, word_embedding, train_files_path, validation_files_path, epoch_start, max_epoch_number, learning_rate):

        global_step = 0
        for epoch in range(epoch_start, max_epoch_number):
            stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            self.sess.run(tf.variables_initializer(stream_vars_valid))
            for batch_number in range(len(train_files_path)):
                global_step += 1

                file = train_files_path[batch_number]
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, pair_indices, pair_gold] = load_data(file)
                if len(doc_word) == 0:
                    print("skip this file (zero length document): {}".format(file))
                    continue
                if np.sum(gold_phrase) == 0:
                    print("skip this file (no phrase): {}".format(file))
                    continue
                if np.sum(pair_gold) <= 1:
                    print("skip this file (no pair): {}".format(file))
                    continue

                current_word_ids = doc_word
                current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

                current_char_ids = doc_char
                current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

                current_doc_phrase_indices = phrase_word
                current_doc_gold_phrases = gold_phrase
                current_doc_phrase_length = phrase_word_len


                current_gold_pair = pair_gold
                posetive_indices = np.squeeze(np.argwhere(current_gold_pair == 1))
                negative_indices = np.array(random.choices(np.squeeze(np.argwhere(current_gold_pair == 0)), k=100*len(posetive_indices)))
                all_indices = np.concatenate([negative_indices, posetive_indices])
                np.random.shuffle(all_indices)

                current_doc_pair_indices = pair_indices[all_indices]
                current_doc_pair_gold = pair_gold[all_indices]

                feed_dict = {
                    self.word_ids: current_word_ids,
                    self.word_embedding: word_embedding,
                    self.sentence_length: current_sentence_length,
                    self.char_ids: current_char_ids,
                    self.word_length: current_word_length,
                    self.phrase_indices: current_doc_phrase_indices,
                    self.gold_phrases: current_doc_gold_phrases,
                    self.phrase_length: current_doc_phrase_length,
                    self.pair_gold: current_doc_pair_gold,
                    self.pair_rep_indices: current_doc_pair_indices,
                    self.dropout_rate: 0.5,
                    self.learning_rate: learning_rate
                }
                try:
                    [_, loss, pred, summary] = self.sess.run([self.pair_identification_train, self.pair_identification_loss
                                                              , self.candidate_pair_probability, self.merged], feed_dict)

                    self.train_writer.add_summary(summary, global_step)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0

                    gold = current_doc_pair_gold

                    precision = precision_score(gold, pred) * 100
                    recall = recall_score(gold, pred) * 100
                    f1_measure = f1_score(gold, pred) * 100
                    try:
                        logger.info("epoch:{:3d} batch:{:4d} loss:{:11.2f} cand:{:7d}/{:8d} gold:{:4d}/{:7d} pred:{:4d}/{:7d} precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}"
                                    .format(epoch, batch_number, loss,
                                            len(all_indices), len(current_gold_pair),
                                            # len(pair_indices), len(pair_indices),
                                            int(np.sum(gold)), len(gold), int(np.sum(pred)), len(pred),
                                            precision, recall, f1_measure))
                    except Exception as e:
                        print("here:{}".format(e))
                        logger.info("epoch:{:3d} batch:{:4d}".format(epoch, batch_number))

                except Exception as e:
                    print("here:{}".format(e))

            save_path = self.saver.save(self.sess, "{}/coref_model".format(self.dir_checkpoint),
                                        global_step=int(epoch), write_meta_graph=False)
            logger.info("model is saved in: {}".format(save_path))
            all_precision = []
            all_recall = []
            all_f1 = []
            stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            self.sess.run(tf.variables_initializer(stream_vars_valid))
            for doc_num in range(len(validation_files_path)):
                file = validation_files_path[doc_num]
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, pair_indices, pair_gold] = load_data(file)
                if len(doc_word) == 0:
                    print("skip this file (zero length document): {}".format(file))
                    continue
                if np.sum(gold_phrase) == 0:
                    print("skip this file (no phrase): {}".format(file))
                    continue


                current_word_ids = doc_word
                current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

                current_char_ids = doc_char
                current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

                current_doc_phrase_indices = phrase_word
                current_doc_gold_phrases = gold_phrase
                current_doc_phrase_length = phrase_word_len

                current_doc_pair_indices = pair_indices
                current_doc_pair_gold = pair_gold

                feed_dict = {
                    self.word_ids: current_word_ids,
                    self.word_embedding: word_embedding,
                    self.sentence_length: current_sentence_length,
                    self.char_ids: current_char_ids,
                    self.word_length: current_word_length,
                    self.phrase_indices: current_doc_phrase_indices,
                    self.gold_phrases: current_doc_gold_phrases,
                    self.phrase_length: current_doc_phrase_length,
                    self.pair_gold: current_doc_pair_gold,
                    self.pair_rep_indices: current_doc_pair_indices,
                    self.dropout_rate: 0
                }
                try:
                    [pred, summary] = self.sess.run([self.candidate_pair_probability, self.merged], feed_dict)

                    self.validation_writer.add_summary(summary, global_step)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0

                    gold = current_doc_pair_gold

                    precision = precision_score(gold, pred) * 100
                    all_precision.append(precision)
                    recall = recall_score(gold, pred) * 100
                    all_recall.append(recall)
                    f1_measure = f1_score(gold, pred) * 100
                    all_f1.append(f1_measure)
                    logger.info("val-{}:{:3d} precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}"
                                .format(file, doc_num, precision, recall, f1_measure))
                except Exception as e:
                    print(e)

            avg_precision = np.average(all_precision)
            avg_recall = np.average(all_recall)
            avg_f1 = np.average(all_f1)

            logger.info("====================>epoch:{:3d} validation metrics: precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}".format(epoch, avg_precision, avg_recall, avg_f1))


    def evaluate_pair_identification(self, word_embedding, test_files_path):

        all_precision = []
        all_recall = []
        all_f1 = []
        for doc_num in range(len(test_files_path)):
            file = test_files_path[doc_num]
            [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, pair_indices, pair_gold] = load_data(file)
            if len(doc_word) == 0:
                print("skip this file (zero length document): {}".format(file))
                continue
            if np.sum(gold_phrase) == 0:
                print("skip this file (no phrase): {}".format(file))
                continue


            current_word_ids = doc_word
            current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)

            current_char_ids = doc_char
            current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)

            current_doc_phrase_indices = phrase_word
            current_doc_gold_phrases = gold_phrase
            current_doc_phrase_length = phrase_word_len

            current_doc_pair_indices = pair_indices
            current_doc_pair_gold = pair_gold

            feed_dict = {
                self.word_ids: current_word_ids,
                self.word_embedding: word_embedding,
                self.sentence_length: current_sentence_length,
                self.char_ids: current_char_ids,
                self.word_length: current_word_length,
                self.phrase_indices: current_doc_phrase_indices,
                self.gold_phrases: current_doc_gold_phrases,
                self.phrase_length: current_doc_phrase_length,
                self.pair_gold: current_doc_pair_gold,
                self.pair_rep_indices: current_doc_pair_indices,
                self.dropout_rate: 0
            }
            try:
                [pred] = self.sess.run([self.candidate_pair_probability], feed_dict)

                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                gold = current_doc_pair_gold

                precision = precision_score(gold, pred) * 100
                all_precision.append(precision)
                recall = recall_score(gold, pred) * 100
                all_recall.append(recall)
                f1_measure = f1_score(gold, pred) * 100
                all_f1.append(f1_measure)
                logger.info("{.3d}/{.3d}- precision:{:6.2f} recall:{:6.2f} f1:{:6.2f} file:{}"
                            .format(doc_num, len(test_files_path), precision, recall, f1_measure, file))
            except Exception as e:
                print(e)

        avg_precision = np.average(all_precision)
        avg_recall = np.average(all_recall)
        avg_f1 = np.average(all_f1)

        logger.info("test metrics: precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}".format(avg_precision, avg_recall, avg_f1))




