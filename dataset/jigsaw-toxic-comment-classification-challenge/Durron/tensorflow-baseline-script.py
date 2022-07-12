# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output

###################################################################
###################################################################
###################################################################
# Used the following script as template:
# https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069
# Translated this to tensorflow
###################################################################
###################################################################
###################################################################

from keras.preprocessing import text, sequence
import tensorflow as tf

max_features = 20000
maxlen = 100

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


# Create model
X_ph = tf.placeholder(tf.int32, [None, maxlen])
y_ph = tf.placeholder(tf.float32, [None, 6])
dropout_ph = tf.placeholder(tf.float32)

embed_size = 128
embedding_layer = tf.get_variable(name="embeddings",
                                  initializer=tf.random_uniform([max_features, embed_size], -1.0, 1.0))

lstm_cell_fw = tf.contrib.rnn.LSTMCell(50)
lstm_cell_bw = tf.contrib.rnn.LSTMCell(50)

input_lookup = tf.nn.embedding_lookup(embedding_layer, X_ph)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,
                                                         lstm_cell_bw,
                                                         inputs=input_lookup,
                                                         dtype=tf.float32,
                                                         time_major=False)

outputs = tf.concat(outputs, 2)
outputs = tf.nn.dropout(tf.keras.layers.GlobalMaxPool1D()(outputs), dropout_ph)
dense_output = tf.nn.dropout(tf.layers.dense(inputs=outputs, units=50, activation=tf.nn.relu), dropout_ph)
dense_output = tf.layers.dense(inputs=dense_output, units=6)

probabilities = tf.nn.sigmoid(dense_output)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y_ph,
    logits=dense_output
)

loss = tf.reduce_mean(cross_entropy)
adam_opt = tf.train.AdamOptimizer().minimize(loss)

val_data_count = int(X_t.shape[0] * 0.1)

train_data_count = X_t.shape[0] - val_data_count
x_train = X_t[:train_data_count, :]
y_train = y[:train_data_count, :]
x_val = X_t[train_data_count:, :]
y_val = y[train_data_count:, :]

epochs = 2
batch_size = 32

# Create session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

# Train model
for epoch in range(epochs):

    data_count = 0
    print("Epoch", epoch)

    while data_count < train_data_count:

        # Train step
        feed_dict = {
            X_ph: x_train[data_count:data_count+batch_size],
            y_ph: y_train[data_count:data_count+batch_size],
            dropout_ph: 0.9
        }

        train_loss, _ = sess.run([loss, adam_opt], feed_dict=feed_dict)

        # Validation step
        feed_dict = {
            X_ph: x_val,
            y_ph: y_val,
            dropout_ph: 1.0
        }
        val_loss = sess.run(loss, feed_dict=feed_dict)

        # Increase data count
        data_count += x_train[data_count:data_count+batch_size].shape[0]
        print(data_count, "/", train_data_count, "val_loss", val_loss, "train_loss", train_loss)

    saver.save(sess, "weights/model.ckpt", global_step=epoch)

saver.restore(sess, tf.train.latest_checkpoint("weights/"))

# Predict test data
test_data_count = X_te.shape[0]

data_count = 0
batch_size = 10000
test_probs = []
while data_count != test_data_count:

    # Prediction step
    feed_dict = {
        X_ph: X_te[data_count:data_count + batch_size],
        dropout_ph: 1.0
    }

    probs = sess.run(probabilities, feed_dict=feed_dict)
    test_probs.append(probs)

    # Increase data count
    data_count += X_te[data_count:data_count + batch_size].shape[0]
    print(data_count, "/", test_data_count)

test_probs = np.concatenate(test_probs)
df_probs = pd.DataFrame(test_probs, columns=list_classes)
result_df = pd.concat([test[["id"]], df_probs], axis=1)
result_df.to_csv("baseline_tf.csv", index=False)