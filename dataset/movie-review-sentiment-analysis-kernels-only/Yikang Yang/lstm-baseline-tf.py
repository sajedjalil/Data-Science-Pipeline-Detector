import tensorflow as tf
import numpy as np
import pandas as pd
import re

train_path = '../input/train.tsv'
test_path = '../input/test.tsv'
word_vec = '../input/glove6b/glove.6B.200d.txt'


#train_path = 'data/train.tsv'
#test_path = 'data/test.tsv'
#word_vec = 'data/glove.6B.100d.txt'

# ---- load training dataset ----
class data_tool(object):

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        # training set
        self.train = pd.read_table(train_path, sep="\t")
        self.train_x = [self.str_clean(i) for i in self.train['Phrase']]

        self.train_y = [[0] * 4 for i in range(self.train.shape[0])]
        _ = [self.train_y[i].insert(j, 1) for i, j in enumerate(self.train['Sentiment'])]
        self.train_y = np.array(self.train_y)

        # test set
        self.test = pd.read_table(test_path, sep='\t')
        self.test_x = [self.str_clean(i) for i in self.test['Phrase']]

        # build corpus
        self.max_length, self.vocab_dict = self.word_corpus(self.train_x + self.test_x)

        # Tokenize, convert text to a list of integers
        self.train_x = self.text2index(self.train_x, self.vocab_dict, self.max_length)
        self.test_x = self.text2index(self.test_x, self.vocab_dict, self.max_length)

    def word_corpus(self, text_x):
        # Tokenize words
        vocab_dict = {word: index + 1 for index, word in enumerate(set(' '.join(text_x).split()))}

        # maximum sequence length
        max_sequence_length = len(max([i.split() for i in text_x], key=len))
        return max_sequence_length, vocab_dict

    def text2index(self, text, vocab_dict, maximum_length):
        """
        tokenization
        """
        text = [i.split() for i in text]
        tmp = np.zeros(shape=(len(text), maximum_length))
        for i in range(len(text)):
            for j in range(len(text[i])):
                tmp[i][j] = vocab_dict.get(text[i][j], 0)
        return tmp

    def str_clean(self, string):
        """
        Tokenization/string cleaning forn all dataset except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
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

    # generate batches of data to train
    def generate_batches(self, data, epoch_size, batch_size, shuffle=False):
        data = np.array(data)

        data_size = len(data)

        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j+1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def save_data(self, result):
        test_data = pd.read_table(test_path, sep='\t')
        test_data['Sentiment'] = result.reshape(-1).tolist()
        test_data = test_data.loc[:, ['PhraseId', 'Sentiment']]
        print(test_data)
        test_data.to_csv("sample_submission.csv", index=False)


# --- build RNN model ----
class TextRNN(object):

    def __init__(self, sequence_length, embedding_size, lstm_size, vocabulary_size, num_classes, word_vec=None):
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size

        self.vocabulary_size = vocabulary_size

        # define placeholders for input and label
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], 'input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], 'labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.real_seq_length = tf.placeholder(tf.float32, [None], name='name_seq_length')

        # embedding
        with tf.name_scope("embedding"):
            if word_vec:
                W = tf.get_variable("embedding_W", dtype=tf.float32,
                                    initializer=word_vec)
            else:
                W = tf.get_variable("embedding_W", dtype=tf.float32,
                                    initializer=tf.truncated_normal(shape=[vocabulary_size, embedding_size], stddev=0.1))

            self.embedded_char = tf.nn.embedding_lookup(W, self.input_x, name='embedded_vectors')
            # self.embedded_char_tr = tf.transpose(self.embedded_char,perm=[1, 0 ,2])

        # RNN
        with tf.name_scope("RNN"):
            self.cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.cell = tf.contrib.rnn.DropoutWrapper(self.cell,
                                                      output_keep_prob=self.keep_prob)
            self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.embedded_char,
                                                          sequence_length=self.real_seq_length,
                                                          dtype=tf.float32)

        # linear transformation
        with tf.name_scope("linear"):
            W = tf.get_variable('output_W', shape=[lstm_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(initial_value=tf.constant([0.1] * num_classes), name='output_bias')
            self.scores = tf.nn.xw_plus_b(self.states[-1], W, b, name='scores')
            self.outputs = tf.argmax(self.scores, axis=1, name='output')

        # loss and accuracy
        with tf.name_scope("loss_accuracy"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.outputs, tf.argmax(self.input_y, axis=1)), "float"))


class Training(data_tool, TextRNN):

    def __init__(self):
        self.epoch_size = 7
        self.batch_size = 128
        print("init data..")
        data_tool.__init__(self, train_path=train_path, test_path=test_path)

        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                print("init model..")
                TextRNN.__init__(self, sequence_length=self.max_length,
                                 embedding_size=200, num_classes=5,
                                 lstm_size=128, vocabulary_size=self.vocab_dict.keys().__len__())

                global_step = tf.Variable(0, name='global_step', trainable=False)

                self.saver = tf.train.Saver()

                optimizer = tf.train.AdamOptimizer(0.001)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step)

                # get real_length
                def real_length(batches):
                    return np.ceil([np.argmin(batch.tolist()+[0]) for batch in batches])

                # initialize variable
                sess.run(tf.global_variables_initializer())

                # generate batches
                batches_all = self.generate_batches(list(zip(self.train_x, self.train_y)), epoch_size=self.epoch_size,
                                                    batch_size=self.batch_size, shuffle=True)
                total_amount = (len(self.train_x) // self.batch_size + 1) * self.epoch_size
                for i, batch in enumerate(batches_all):
                    batch_x, batch_y = zip(*batch)
                    loss, _, accuracy, step = sess.run([self.loss, train_op, self.accuracy, global_step],
                                                       feed_dict={self.input_x: batch_x,
                                                                  self.input_y: batch_y,
                                                                  self.keep_prob: 0.2,
                                                                  self.real_seq_length: real_length(batch_x)})
                    print("Currently at batch {}/{}".format(i, total_amount), "The loss is %f" % loss)
                    if i % 100 == 0:
                        print("current batch accuracy is:", accuracy)
                        self.saver.save(sess, "/tmp/model1.ckpt", global_step=i)

                # start testing training
                data_size = len(self.test_x)
                result = []
                for i in range(data_size // 500):
                    tmp = self.test_x[i * 500:(i + 1) * 500]
                    result.append(sess.run(self.outputs, feed_dict={self.input_x: tmp,
                                                                    self.keep_prob: 1.0,
                                                                    self.real_seq_length: real_length(tmp)}
                                           ))
                tmp = self.test_x[(i+1)*500:]
                result.append(sess.run(self.outputs, feed_dict={self.input_x: self.test_x[(i + 1) * 500:],
                                                                self.keep_prob: 1.0,
                                                                self.real_seq_length: real_length(tmp)}))
                self.result = np.concatenate(result, axis=0)

                self.save_data(self.result)

if __name__ == '__main__':
    train_ = Training()
