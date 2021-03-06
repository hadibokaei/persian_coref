import tensorflow as tf
from  common.utility import pad_sequences, logger, load_data
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import random
from common.utility import convert_pairs_to_clusters


class CorefModel(object):

    def __init__(self, word_vocab_size, char_vocab_size, word_embedding_dimension, char_embedding_dimension, max_word_length, max_phrase_length
                 , conv_filter_num, conv_filter_size
                 , lstm_unit_size, dir_tensoboard_log, dir_checkpoint, keep_phrase_ratio):
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
        self.keep_phrase_ratio = keep_phrase_ratio

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
        self.word_ids           = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")         #shape=[# of sentences in doc, max # of words in sentences]
        self.word_embedding     = tf.placeholder(tf.float32
                , shape=[self.word_vocab_size, self.word_embedding_dimension], name="word_embedding")   #shape=[vocab size, embedding dimension]
        self.sentence_length    = tf.placeholder(tf.int32, shape=[None], name="sentence_length")        #shape=[# of sentences in doc]
        self.char_ids           = tf.placeholder(tf.int32, shape=[None, None, self.max_word_length])    #shape=[# of sentences in doc, max # of words in sentences, max # of characters in a word]
        self.word_length        = tf.placeholder(tf.int32, shape=[None, None], name="word_length")      #shape=[# of sentences in doc, max # of words in sentences]
        self.phrase_indices     = tf.placeholder(tf.int32, shape=[None, self.max_phrase_length, 2])     #shape=[# of candidate phrases in doc, max phrase length in words, 2]
        self.gold_phrases       = tf.placeholder(tf.int32, shape=[None])                                #shape=[# of candidate phrases in doc]
        self.phrase_length      = tf.placeholder(tf.int32, shape=[None])                                #shape=[# of candidate phrases in doc]
        self.pair_gold          = tf.placeholder(tf.int32, shape=[None, 2])                             #shape=[# of pairs in doc, 2]
        self.dropout_rate       = tf.placeholder(tf.float32, shape=[])                                  #scalar
        self.learning_rate      = tf.placeholder(tf.float32, shape=[])                                  #scalar


    def add_word_representation(self):
        embedded_words = tf.nn.embedding_lookup(self.word_embedding, self.word_ids, name="embedded_words")              #shape=[# of sentences in doc, max # of words in sentences, word embedding dimension]
        char_embedding = tf.get_variable(dtype=tf.float32, shape=[self.char_vocab_size, self.char_embedding_dimension], name="char_embeddings")
        embedded_chars = tf.nn.embedding_lookup(char_embedding, self.char_ids, name='embedded_chars')                   #shape=[# of sentences in doc, max # of words in sentences, max number of characters in a word, char embedding dimension]
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

    def add_fcn_phrase(self):

        dropped_rep = tf.keras.layers.Dropout(rate = self.dropout_rate)(self.phrase_rep)
        dense_output = tf.keras.layers.Dense(self.lstm_unit_size,activation='elu')(dropped_rep) # shape = [# of candidate phrases, lstm hidden size]
        tf.summary.histogram("output layer", dense_output)
        dropped_dense_output = tf.keras.layers.Dropout(rate = self.dropout_rate)(dense_output)
        self.candidate_phrase_logit = tf.squeeze(tf.keras.layers.Dense(1)(dropped_dense_output)) # shape = [# of candidate phrases]
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

        k = tf.cast(tf.shape(self.phrase_rep)[0] / self.keep_phrase_ratio, tf.int32)
        indices = tf.math.top_k(self.candidate_phrase_logit, k=k).indices
        indices = tf.concat([indices,[0]], axis = 0)
        y = tf.zeros(tf.shape(indices), dtype=tf.int32)

        z1 = tf.stack([indices, y], axis=1)
        z1f = tf.expand_dims(z1, 0)
        z2 = tf.reverse(z1, axis=[1])
        z2f = tf.expand_dims(z2, 1)
        zf = z1f + z2f
        zf = tf.reverse(zf, axis=[2])


        selected_pairs = zf                                             #shape = [num_of_pruned_phrases, num_of_pruned_phrases, 2]
        selected_pairs_ = tf.expand_dims(selected_pairs, 3)             #shape = [num_of_pruned_phrases, num_of_pruned_phrases, 2, 1]


        self.pair_rep = tf.gather_nd(self.phrase_rep, selected_pairs_)
        self.pair_rep = tf.reshape(self.pair_rep
                        , shape=[tf.shape(self.pair_rep)[0], tf.shape(self.pair_rep)[1], 4*self.lstm_unit_size])    # shape = [num_of_pruned_phrases, num_of_pruned_phrases, 4 * lstm hidden size]
        self.pair_indices = selected_pairs                                                                          # shape = [num_of_pruned_phrases, num_of_pruned_phrases,  2]
        self.indices = indices

    def add_fcn_pair(self):
        dropped_rep = tf.keras.layers.Dropout(rate = self.dropout_rate)(self.pair_rep)
        dense_output = tf.keras.layers.Dense(self.lstm_unit_size,activation='elu')(dropped_rep) # shape = [# of pruned candidate pairs, lstm hidden size]
        self.candidate_pair_logit = tf.squeeze(tf.keras.layers.Dense(1)(dense_output))              # shape = [num_of_pruned_phrases, num_of_pruned_phrases]
        self.candidate_pair_probability = tf.keras.layers.Softmax()(self.candidate_pair_logit)      # shape = [num_of_pruned_phrases, num_of_pruned_phrases]

        self.predicted_pairs = tf.gather_nd(self.pair_indices
                                            ,tf.stack([tf.range(0,tf.shape(self.candidate_pair_probability)[0])
                                            , tf.cast(tf.arg_max(self.candidate_pair_probability,1), tf.int32)],1))

    def add_phrase_loss_train(self):

        self.phrase_identification_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.gold_phrases, tf.float32), logits=self.candidate_phrase_logit))

        # self.phrase_identification_loss = -tf.reduce_mean(tf.math.log(tf.where(self.gold_phrases>0
        #                                                                       , self.candidate_phrase_probability
        #                                                                       , 1-self.candidate_phrase_probability)))
        # tf.summary.scalar("phrase loss", self.phrase_identification_loss)

        self.phrase_identification_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.phrase_identification_loss)

    def add_pair_loss_train(self):

        a = tf.expand_dims(self.pair_indices, 2)
        b = tf.expand_dims(self.pair_gold, 0)
        c = tf.reduce_sum(tf.abs(a - b), -1)

        out1 = tf.reduce_min(c, 2)
        out2 = tf.cast(tf.equal(out1, 0), tf.int32)
        found_indices_ = tf.reduce_sum(out2, 1)
        found_indices = tf.where(tf.math.greater(found_indices_, 0))

        final_logits = tf.gather_nd(self.candidate_pair_logit, found_indices)
        final_gold_dist = tf.gather_nd(out2, found_indices)
        self.pair_indices = tf.gather_nd(self.pair_indices, found_indices)
        final_gold_dist = final_gold_dist/tf.expand_dims(tf.reduce_sum(final_gold_dist,1),1)

        self.pair_identification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=final_gold_dist, logits=final_logits))


        self.pair_identification_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.pair_identification_loss)
        self.final_logits = final_logits
        self.final_gold_dist = final_gold_dist


    def add_final_train(self):
        self.final_loss = 5*self.phrase_identification_loss + self.pair_identification_loss
        # self.final_loss = self.pair_identification_loss
        self.final_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.final_loss)

    def train_pair_identification(self, word_embedding, train_files_path, validation_files_path, epoch_start, max_epoch_number, learning_rate):

        global_step = 0
        for epoch in range(epoch_start, max_epoch_number):
            stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            self.sess.run(tf.variables_initializer(stream_vars_valid))
            for batch_number in range(len(train_files_path)):
                global_step += 1

                file = train_files_path[batch_number]
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, gold_phrase_id_pair, clusters, gold_2_local_phrase_id_map] = load_data(file)
                if len(doc_word) == 0:
                    print("{} skippend".format(file))
                    continue

                #اضافه کردن عبارت خالی
                phrase_word = np.concatenate((np.expand_dims(np.zeros_like(phrase_word[0]), axis=0), phrase_word))
                phrase_word_len = np.concatenate(([0],phrase_word_len))
                gold_phrase = np.concatenate(([0],gold_phrase))
                gold_2_local_phrase_id_map[0] = -1

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

                # current_doc_pair_gold = [[gold_2_local_phrase_id_map[a]+1 for a in x if a in gold_2_local_phrase_id_map.keys()] for x in gold_phrase_id_pair]

                # z = [[[gold_2_local_phrase_id_map[x]+1, gold_2_local_phrase_id_map[y]+1]
                #       for x in a for y in a
                #       if x in gold_2_local_phrase_id_map.keys() and y in gold_2_local_phrase_id_map.keys() and x!=y and x!=0 and x>y]
                #      for a in clusters]


                z_ = [list(x) for x in clusters]
                z = [[[gold_2_local_phrase_id_map[y[x+1]]+1,gold_2_local_phrase_id_map[y[x]]+1]
                      for x in range(len(y)-1) if y[x+1] in gold_2_local_phrase_id_map.keys() and y[x] in gold_2_local_phrase_id_map.keys()] for y in z_]

                current_doc_pair_gold = []
                for x in z:
                    current_doc_pair_gold += x

                # current_gold_pair = pair_gold
                # posetive_indices = np.squeeze(np.argwhere(current_gold_pair == 1))
                # negative_indices = np.array(random.choices(np.squeeze(np.argwhere(current_gold_pair == 0)), k=100*len(posetive_indices)))
                # all_indices = np.concatenate([negative_indices, posetive_indices])
                # np.random.shuffle(all_indices)
                #
                # current_doc_pair_indices = pair_indices[all_indices]
                # current_doc_pair_gold = pair_gold[all_indices]

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
                    self.dropout_rate: 0.5,
                    self.learning_rate: learning_rate
                }
                try:
                    [_, loss, pair_probability, pair_indices, summary, predicted_pairs, indices,pair_rep, final_logits, final_gold_dist, phrase_rep] = \
                        self.sess.run([self.final_train
                                          , self.final_loss
                                          , self.candidate_pair_probability
                                          , self.pair_indices
                                          , self.merged
                                          , self.predicted_pairs
                                          , self.indices, self.pair_rep, self.final_logits, self.final_gold_dist, self.phrase_rep
                                       ], feed_dict)

                    # print(np.shape(pair_indices))

                    predicted_clusters = convert_pairs_to_clusters(predicted_pairs)
                    gold_clusters = [[gold_2_local_phrase_id_map[x] + 1 for x in a if x in gold_2_local_phrase_id_map.keys()] for a  in clusters]

                    # print(predicted_pairs)
                    print(predicted_clusters)
                    print(gold_clusters)

                    # [_, loss, pred, summary] = self.sess.run([self.pair_identification_train, self.pair_identification_loss
                    #                                           , self.candidate_pair_probability, self.merged], feed_dict)

                    # self.train_writer.add_summary(summary, global_step)
                    # pred[pred > 0.5] = 1
                    # pred[pred <= 0.5] = 0
                    #
                    # gold = current_doc_pair_gold

                    # precision = precision_score(gold, pred) * 100
                    # recall = recall_score(gold, pred) * 100
                    # f1_measure = f1_score(gold, pred) * 100
                    try:
                        # logger.info("epoch:{:3d} batch:{:4d} loss:{:11.2f} precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}"
                        #             .format(epoch, batch_number, loss, precision, recall, f1_measure))
                        logger.info("epoch:{:3d} batch:{:4d} loss:{:11.2f} ({})"
                                    .format(epoch, batch_number, loss, file))
                    except Exception as e:
                        print("here:{}".format(e))
                        logger.info("epoch:{:3d} batch:{:4d}".format(epoch, batch_number))

                except Exception as e:
                    print("here:{}".format(e))

            save_path = self.saver.save(self.sess, "{}/coref_model".format(self.dir_checkpoint),
                                        global_step=int(epoch), write_meta_graph=False)
            logger.info("model is saved in: {}".format(save_path))
            # all_precision = []
            # all_recall = []
            # all_f1 = []
            # stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            # self.sess.run(tf.variables_initializer(stream_vars_valid))
            # for doc_num in range(len(validation_files_path)):
            #     file = validation_files_path[doc_num]
            #     [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, pair_indices, pair_gold] = load_data(file)
            #     if len(doc_word) == 0:
            #         print("skip this file (zero length document): {}".format(file))
            #         continue
            #     if np.sum(gold_phrase) == 0:
            #         print("skip this file (no phrase): {}".format(file))
            #         continue
            #
            #
            #     current_word_ids = doc_word
            #     current_word_ids, current_sentence_length = pad_sequences(current_word_ids, 0)
            #
            #     current_char_ids = doc_char
            #     current_char_ids, current_word_length = pad_sequences(current_char_ids, 0, nlevels=2)
            #
            #     current_doc_phrase_indices = phrase_word
            #     current_doc_gold_phrases = gold_phrase
            #     current_doc_phrase_length = phrase_word_len
            #
            #     current_doc_pair_indices = pair_indices
            #     current_doc_pair_gold = pair_gold
            #
            #     feed_dict = {
            #         self.word_ids: current_word_ids,
            #         self.word_embedding: word_embedding,
            #         self.sentence_length: current_sentence_length,
            #         self.char_ids: current_char_ids,
            #         self.word_length: current_word_length,
            #         self.phrase_indices: current_doc_phrase_indices,
            #         self.gold_phrases: current_doc_gold_phrases,
            #         self.phrase_length: current_doc_phrase_length,
            #         # self.pair_gold: current_doc_pair_gold,
            #         self.dropout_rate: 0
            #     }
            #     try:
            #         [pred, summary] = self.sess.run([self.candidate_pair_probability, self.merged], feed_dict)
            #
            #         self.validation_writer.add_summary(summary, global_step)
            #         pred[pred > 0.5] = 1
            #         pred[pred <= 0.5] = 0
            #
            #         gold = current_doc_pair_gold
            #
            #         precision = precision_score(gold, pred) * 100
            #         all_precision.append(precision)
            #         recall = recall_score(gold, pred) * 100
            #         all_recall.append(recall)
            #         f1_measure = f1_score(gold, pred) * 100
            #         all_f1.append(f1_measure)
            #         logger.info("val-{}:{:3d} precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}"
            #                     .format(file, doc_num, precision, recall, f1_measure))
            #     except Exception as e:
            #         print(e)
            #
            # avg_precision = np.average(all_precision)
            # avg_recall = np.average(all_recall)
            # avg_f1 = np.average(all_f1)
            #
            # logger.info("====================>epoch:{:3d} validation metrics: precision:{:6.2f} recall:{:6.2f} f1:{:6.2f}".format(epoch, avg_precision, avg_recall, avg_f1))


    def evaluate_pair_identification(self, word_embedding, test_files_path):

        all_precision = []
        all_recall = []
        all_f1 = []
        for doc_num in range(len(test_files_path)):
            file = test_files_path[doc_num]
            [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, gold_phrase_id_pair, clusters,
             gold_2_local_phrase_id_map] = load_data(file)
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

            z = [[[gold_2_local_phrase_id_map[x], gold_2_local_phrase_id_map[y]] for x in a for y in a if
                  x < y and x in gold_2_local_phrase_id_map.keys() and y in gold_2_local_phrase_id_map.keys()] for a
                 in clusters]

            current_doc_pair_gold = []
            for x in z:
                current_doc_pair_gold += x

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


    def evaluate_phrase_identification(self, word_embedding, test_files_path):
        all_precision = []
        all_recall = []
        all_f1 = []
        stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
        self.sess.run(tf.variables_initializer(stream_vars_valid))
        for doc_num in range(len(test_files_path)):

            file = test_files_path[doc_num]
            [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, _, _, _] = load_data(file)

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

            print("+{}-{}:{}".format(np.sum(gold_phrase), len(gold_phrase) - np.sum(gold_phrase), len(gold_phrase)))

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
                [pred] = self.sess.run([self.candidate_phrase_probability], feed_dict)

                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                gold = current_doc_gold_phrases

                precision = precision_score(gold, pred) * 100
                all_precision.append(precision)
                recall = recall_score(gold, pred) * 100
                all_recall.append(recall)
                f1_measure = f1_score(gold, pred) * 100
                all_f1.append(f1_measure)
                logger.info("{}/{}: precision:{:5.2f} recall:{:5.2f} f1:{:5.2f} ({})".format(doc_num, len(test_files_path), precision, recall, f1_measure, file))
            except Exception as e:
                print(e)

        avg_precision = np.average(all_precision)
        avg_recall = np.average(all_recall)
        avg_f1 = np.average(all_f1)

        logger.info("precision:{:5.2f} recall:{:5.2f} f1:{:5.2f}".format(avg_precision, avg_recall, avg_f1))

    def train_phrase_identification(self, word_embedding, train_files_path, validation_files_path, epoch_start, max_epoch_number, learning_rate):
        global_step = 0
        for epoch in range(epoch_start, max_epoch_number):
            stream_vars_valid = [v for v in tf.local_variables() if 'metrics/' in v.name]
            self.sess.run(tf.variables_initializer(stream_vars_valid))
            for batch_number in range(len(train_files_path)):

                global_step += 1

                file = train_files_path[batch_number]
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, _, _, _] = load_data(file)
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

                current_gold_phrase = np.array(gold_phrase)
                current_phrase_word = np.array(phrase_word)
                current_phrase_word_len = np.array(phrase_word_len)
                num_posetive = np.sum(current_gold_phrase)
                num_negatives = len(current_gold_phrase) - num_posetive
                k = num_negatives
                # negative_indices = np.array(random.choices(np.squeeze(np.argwhere(current_gold_phrase == 0)), k=k))
                negative_indices = np.squeeze(np.argwhere(current_gold_phrase == 0))
                posetive_indices = np.array(random.choices(np.squeeze(np.argwhere(current_gold_phrase == 1)), k=k))
                # posetive_indices = np.squeeze(np.argwhere(current_gold_phrase == 1))
                all_indices = np.concatenate([negative_indices, posetive_indices])
                np.random.shuffle(all_indices)
                print("+{}-{}:{}/{}".format(len(posetive_indices), len(negative_indices), len(all_indices), len(current_gold_phrase)))
                current_doc_phrase_indices = current_phrase_word[all_indices]
                current_doc_gold_phrases = current_gold_phrase[all_indices]
                current_doc_phrase_length = current_phrase_word_len[all_indices]

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
                    [pred, _, loss, summary] = self.sess.run([self.candidate_phrase_probability, self.phrase_identification_train, self.phrase_identification_loss, self.merged], feed_dict)
                    # [pred] = self.sess.run([self.candidate_phrase_probability], feed_dict)

                    # self.train_writer.add_summary(summary, global_step)
                    pred[pred > 0.5] = 1
                    pred[pred <= 0.5] = 0

                    gold = np.array(current_doc_gold_phrases)

                    pred_indices = np.where(pred == 1)
                    gold_indices = np.where(gold == 1)


                    # print(pred_indices)
                    # print(np.shape(pred_indices))
                    # print(gold_indices)
                    # print(np.shape(gold_indices))

                    precision = precision_score(gold, pred) * 100
                    recall = recall_score(gold, pred) * 100
                    f1_measure = f1_score(gold, pred) * 100
                    logger.info("epoch:{:3d} batch:{:4d} loss:{:5.3f} precision:{:5.2f} recall:{:5.2f} f1:{:5.2f} ({})"
                                .format(epoch, batch_number, loss, precision, recall, f1_measure, file))
                    # logger.info("epoch:{:3d} batch:{:4d} loss:{:5.3f} precision:{:5.2f} recall:{:5.2f} f1:{:5.2f} ({})"
                    #             .format(epoch, batch_number, 0, precision, recall, f1_measure, file))
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
                [doc_word, doc_char, phrase_word, phrase_word_len, gold_phrase, _, _, _] = load_data(file)

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



