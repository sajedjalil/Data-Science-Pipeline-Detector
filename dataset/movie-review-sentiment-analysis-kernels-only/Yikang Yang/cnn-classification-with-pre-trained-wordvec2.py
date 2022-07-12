import pandas as pd
import re
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

train_path = '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'
test_path = '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'
word_vec = '../input/glove6b/glove.6B.200d.txt'

# train_path = 'data/train.tsv'
# test_path = 'data/test.tsv'
# word_vec = 'data/glove.6B.100d.txt'

# data_tool
class word_vector(object):
    """
    load pre-trained word embeddings
    """

    def __init__(self, file_path, max_sentence_length=4):
        self.file_path = file_path

        with open(self.file_path, 'r') as f:
            text = f.readlines()

        self.words_dict, self.data = {}, []
        for i, j in enumerate(map(lambda x: x.split(' '), text)):
            self.words_dict[j[0]] = i + 1
            self.data.append(list(map(float, j[1:])))

        # add an index for words that are not in word set.
        self.vocab_size = len(self.words_dict) + 1
        self.words_dict['<UNK>'] = 0

        # add a row of 0 for words that are not in word set.
        self.embedding_size = len(self.data[0])

        # self.data.append([0]*self.embedding_size)
        self.data = np.array(self.data)
        self.data = np.concatenate([np.zeros([1, self.embedding_size]), self.data], axis=0)

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_sentence_length,
                                                                       tokenizer_fn=self.tokenize,
                                                                       vocabulary=self.words_dict)

    def tokenize(self, iterator):
        for i in iterator:
            lis = []
            for j in i.split(' '):
                if j not in self.vocab_processor.vocabulary_:
                    j = "<UNK>"
                lis.append(j)
            yield lis


def clean_str(string):
    '''
    Tokenization/string cleaning forn all dataset except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    return string
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(data_path):
    data = pd.read_table(filepath_or_buffer=data_path, sep='\t')
    sentence_x = data['Phrase']
    sentiment_y = data['Sentiment']

    # clean text
    x_data = [clean_str(i) for i in sentence_x]

    # generate one-hot vector for labels
    y_label = [[0] * 4 for i in sentiment_y]
    _ = [y_label[i].insert(j, 1) for i, j in enumerate(sentiment_y)]
    return x_data, y_label


def load_test_data(data_path):
    data = pd.read_table(filepath_or_buffer=data_path, sep='\t')

    sentence_x = data['Phrase']

    # clean text
    x_data = [clean_str(i) for i in sentence_x]
    return x_data


def batches_generate(data, epoch_size=150, batch_size=64, shuffle=True):
    """
        Generates a batch iterator for a dataset.
        There will be epoch_size * num_batch_per_epoch batches in total
    """
    data = np.array(data)

    # records of data
    data_size = len(data)

    # batches per epoch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(epoch_size):
        # Shuffle the data ata each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]


# model
class TextCNN(object):
    """
    input_x: placeholder, sequence of integers that represent sentences
    input_y: placeholder, a one-hot vector that represent label
    """

    def __init__(self, sequence_length, embedding_size, word_vector, filter_sizes, num_filters):
        # basic properties:
        self.sequence_lenth = sequence_length
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # define placeholders
        self.input_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, 5], name='label_y')
        self.keep_prob = tf.placeholder(tf.float32, name='Dropout_keep_prob')

        # word embeddings
        with tf.name_scope('embeddings'):
            W = tf.get_variable('W', initializer=tf.constant(word_vector, dtype=tf.float32)) 
            self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_chars')
            self.embedded_char_expanded = tf.expand_dims(self.embedded_char, axis=-1)

        # cnn with multi-filters and pooling
        pooling_output = []
        for i, filter_size in enumerate(filter_sizes):
            pool = self.cnn(input=self.embedded_char_expanded, filter_size=filter_size, index=i)
            pooling_output.append(pool)

        # flatten all pooling output
        self.pool = tf.concat(pooling_output, axis=-1)
        total_num_neorons = num_filters * len(filter_sizes)

        self.pool = tf.reshape(self.pool, shape=[-1, total_num_neorons])


        # add dropout:
        with tf.name_scope('Dropout'):
            self.drop_out = tf.nn.dropout(self.pool, keep_prob=self.keep_prob)


        with tf.name_scope('fully_connnected'):
            self.full_connect1 = fully_connected(self.drop_out, num_outputs=500, activation_fn=tf.nn.relu)
            self.full_connect1_dropout = tf.nn.dropout(self.full_connect1, keep_prob=self.keep_prob)
            self.full_connect2 = fully_connected(self.full_connect1_dropout, num_outputs=500, activation_fn=tf.nn.relu)
            self.full_connect2 = tf.nn.dropout(self.full_connect2, keep_prob=self.keep_prob)


        l2_loss = tf.constant(0.0)
        # get output
        with tf.name_scope('output'):
            W = tf.get_variable('output_W', shape=[500, 5],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(initial_value=tf.constant([0.1] * 5), name='output_bias')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.full_connect2, W, b, name='scores')
            self.output = tf.argmax(self.scores, axis=1, name='output')

        # compute loss and accuracy
        with tf.name_scope('loss_and_accuracy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(loss) + 0.001 * l2_loss
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self.input_y, axis=1)), "float"))

    def cnn(self, input, filter_size, index):
        with tf.name_scope('cnn_maxpool_%s' % index):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.get_variable(name="cnn_Weight_%s" % filter_size,
                                initializer=tf.truncated_normal(shape=filter_shape, stddev=0.1))

            b = tf.get_variable(name="cnn_bias_%s" % filter_size,
                                initializer=tf.constant(0.1, shape=[self.num_filters]))
            # convolutional layer
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="VALID", name='conv')

            # add bias and apply non-linearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')

            # apply max_pooling
            return tf.nn.max_pool(h, ksize=[1, self.sequence_lenth - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                  padding='VALID')


# train
def save_data(sess, test_x, model):
    data_size = len(test_x)
    result = []
    for i in range(data_size // 500):
        result.append(sess.run(model.output, feed_dict={model.input_x: test_x[i * 500:(i + 1) * 500], model.keep_prob: 1.0}))
    result.append(sess.run(model.output, feed_dict={model.input_x: test_x[(i + 1) * 500:], model.keep_prob: 1.0}))
    result = np.concatenate(result, axis=0)

    test_data = pd.read_table(test_path, sep='\t')
    test_data['Sentiment'] = result.reshape(-1).tolist()
    test_data = test_data.loc[:, ['PhraseId', 'Sentiment']]
    print(test_data)
    test_data.to_csv("sample_submission.csv", index=False)

def generate_data():
    train_x, train_y = load_data(train_path)
    test_x = load_test_data(test_path)

    # generate integer vectors
    max_seq_length = len(max(train_x + test_x, key=len))
    word_processor = word_vector(word_vec, max_seq_length)

    train_x_all = np.array(list(word_processor.vocab_processor.transform(train_x)), dtype=np.float32)
    test_x = np.array(list(word_processor.vocab_processor.transform(test_x)), dtype=np.int32)

    train_y = np.array(train_y, dtype=np.float32)

    # shuffle train data
    np.random.seed(100)
    shuffle_index = np.random.permutation(np.arange(len(train_y)))
    train_x = train_x_all[shuffle_index]
    train_y = train_y[shuffle_index]

    print("Vocabulary size: {}".format(len(word_processor.vocab_processor.vocabulary_)))
    return train_x, train_y, word_processor, test_x, train_x_all


def train(train_x, train_y, word_processor, test_x, train_x_all):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNN(sequence_length=word_processor.vocab_processor.max_document_length,
                          embedding_size=word_processor.embedding_size,
                          word_vector=word_processor.data,
                          filter_sizes=[3, 4, 5, 6, 7],
                          num_filters=256)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize
            sess.run(tf.global_variables_initializer())

            # generate batches
            data_train = zip(train_x, train_y)
            batches_train = batches_generate(list(data_train), epoch_size=10, batch_size=64, shuffle=True)
            
            total = (len(train_x)//64+1)*8
            # training
            for i, batch in enumerate(batches_train):
                batch_x, batch_y = zip(*batch)
                loss, _, accuracy = sess.run([cnn.loss, train_op, cnn.accuracy], feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.keep_prob: 0.5})
                print("Currently at batch {}/{}".format(i, total), "The loss is %f" % loss)
                if i % 100 == 0:
                    print("current batch accuracy is:", accuracy)


            save_data(sess, test_x, cnn)


def main(argv=None):
    train_x, train_y, word_processor, test_x, train_x_all = generate_data()
    train(train_x, train_y, word_processor, test_x, train_x_all)


if __name__ == '__main__':
    main()
