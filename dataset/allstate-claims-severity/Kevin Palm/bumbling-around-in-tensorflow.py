import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf


def read_data():
    """Read in the raw datasets and set the index"""

    train = pd.read_csv("../input/train.csv").set_index("id")
    test = pd.read_csv("../input/test.csv").set_index("id")

    return train, test


def column_check(df1, df2):
    """Remove any columns that don't exist in both datasets"""

    for column in df1.columns.values:
        if column not in df2.columns.values:
            df1 = df1.drop(column, axis=1)
    for column in df2.columns.values:
        if column not in df1.columns.values:
            df2 = df2.drop(column, axis=1)

    return df1, df2


def nnet_predict(trX, trY, teX, teY=None):
    """Get predictions using a dropout deep neural network
       I started from nlintz modern net tensorflow example and tried to adapt it.
       https://github.com/nlintz/TensorFlow-Tutorials/blob/master/04_modern_net.py
    """
    

    # Fix label shape
    trY = np.reshape(np.array(trY), [len(trY), 1])

    def init_weights(shape):  # helper function for initializing weights
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):  # defining the network

        # Dropout on input layer, flow to hidden layer 1
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))

        # Dropout on hidden layer 1, flow to hidden layer 2
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))

        # Dropout on hidden layer 2, return output
        h2 = tf.nn.dropout(h2, p_keep_hidden)

        return tf.matmul(h2, w_o)

    # Initialize placeholders for input features and output values
    X = tf.placeholder("float", [None, 1079])
    Y = tf.placeholder("float", [None, 1])

    # Initialize variables for weights
    w_h = init_weights([1079, 540])
    w_h2 = init_weights([540, 540])
    w_o = init_weights([540, 1])

    # Initialize placeholders for dropout parameters
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    # Name model
    py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

    # Define cost as mean absolute error
    cost = tf.reduce_mean(tf.abs(tf.sub(py_x, Y)))

    # Train to minimize mean absolute error using AdamOptimizer
    train_op = tf.train.AdamOptimizer().minimize(cost)

    # Launch the graph in a session
    with tf.Session() as sess:

        # you need to initialize all variables
        tf.initialize_all_variables().run()

        # Count so we can report how many trainings we've finished
        count = 0

        # Set how many trainings we'll do
        for i in range(10):
            count += 1

            # Divide the data into batches for memory sake
            batchsize = int(len(trX)/100)
            for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX) + 1, batchsize)):

                # Train on the batch
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_input: 0.8, p_keep_hidden: 0.5})

            # print("Training {} complete".format(count))

            # If true outcomes were supplied (as in a local cross validation test), go ahead and report loss too
            if teY is not None:
                predicts = sess.run(py_x, feed_dict={X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0})
                print("Mean Absolute Error for round was {}.".format(mean_absolute_error(teY, predicts)))

        # Get final predictions
        predicts = sess.run(py_x, feed_dict={X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0})

        return predicts


def main():

    # Read in the data
    train, test = read_data()

    # Pull out the outcomes
    train_y = train["loss"]
    train_x = train.drop("loss", axis=1)

    # Get dummies and check for misfit dummies
    train_x = pd.get_dummies(train_x)
    test_x = pd.get_dummies(test)
    train_x, test_x = column_check(train_x, test_x)

    test_x["loss"] = nnet_predict(train_x, train_y, test_x)
    test_x[["loss"]].to_csv("nn_predict.csv")


if __name__ == '__main__':
    main()