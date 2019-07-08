import tensorflow as tf

class CorefModel(object):

    def __init__(self, word_vocab_size, char_vocab_size, word_embedding_dimension, char_embedding_dimension, max_word_length
                 , conv_filter_num, conv_filter_size
                 , lstm_unit_size):
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embedding_dimension = word_embedding_dimension
        self.char_embedding_dimension = char_embedding_dimension
        self.max_word_length = max_word_length
        self.conv_filter_num = conv_filter_num
        self.conv_filter_size = conv_filter_size
        self.lstm_unit_size = lstm_unit_size

    def build_graph(self):
        self.add_placeholders()
        self.add_word_representation()
        self.add_lstm()

    def add_placeholders(self):
        self.word_ids           = tf.placeholder(tf.int32, shape=[None, None], name="word_ids") #shape=[# of sentences in doc, max # of words in sentences]
        self.word_embedding     = tf.placeholder(tf.float32, shape=[self.word_vocab_size, self.word_embedding_dimension], name="word_embedding") #shape=[vocab size, embedding dimension]
        self.sentence_length    = tf.placeholder(tf.int32, shape=[None], name="sentence_length") #shape=[# of sentences in doc]
        self.char_ids           = tf.placeholder(tf.int32, shape=[None, None, self.max_word_length]) #shape=[# of sentences in doc, max # of words in sentences, max number of characters in a word]
        self.word_length        = tf.placeholder(tf.int32, shape=[None, None], name="word_length") #shape=[# of sentences in doc, max # of words in sentences]
        self.clusters           = tf.placeholder(tf.int32, shape=[None, None], name="gold_clusters") #shape=[# of clusters in doc, # of spans in clusters]

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
        cell_fw = tf.keras.layers.LSTM(units=self.lstm_unit_size, activation='relu', return_sequences=True)
        cell_bw = tf.keras.layers.LSTM(units=self.lstm_unit_size, activation='relu', return_sequences=True, go_backwards=True)
        self.lstm_output = tf.keras.layers.Bidirectional(layer=cell_fw, backward_layer=cell_bw, merge_mode='concat')(self.word_representation)

        print(self.lstm_output)








