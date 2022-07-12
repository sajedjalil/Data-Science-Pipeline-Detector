import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

###################
# Load & process data
###################
def get_data():
    #################
    # read datasets
    #################
    train = pd.read_csv('../input/train.csv')
    test_submit = pd.read_csv('../input/test.csv')

    # Get y and ID
    train = train[train.y < 250] # Optional: Drop y outliers
    y_train = train['y']
    train = train.drop('y', 1)
    test_submit_id = test_submit['ID']

    #########################
    # Create data
    #########################
    features = ['X0',
                'X5',
                'X118',
                'X127',
                'X47',
                'X315',
                'X311',
                'X179',
                'X314',
                'X232',
                'X29',
                'X263',
                'X261']

    # Build a new dataset using key parameters, lots of drops
    train = train[features]
    test_submit = test_submit[features]

    # Label encoder
    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values) + list(test_submit[c].values))
            train[c] = lbl.transform(list(train[c].values))
            test_submit[c] = lbl.transform(list(test_submit[c].values))

    # Convert to matrix
    train = train.as_matrix()
    y_train = np.transpose([y_train.as_matrix()])
    test_submit = test_submit.as_matrix()
    test_submit_id = test_submit_id.as_matrix()

    #print(train.shape)
    #print(test_submit.shape)

    return train, y_train, test_submit, test_submit_id

#####################
# Neural Network
#####################
# Training steps
STEPS = 500
LEARNING_RATE = 0.0001
BETA = 0.01
DROPOUT = 0.5
RANDOM_SEED = 12345
MAX_Y = 250
RESTORE = True
START = 0

# Training variables
IN_DIM = 13

# Network Parameters - Hidden layers
n_hidden_1 = 100
n_hidden_2 = 50

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)

def deep_network(inputs, keep_prob):
    # Input -> Hidden Layer
    w1 = weight_variable([IN_DIM, n_hidden_1])
    b1 = bias_variable([n_hidden_1])
    # Hidden Layer -> Hidden Layer
    w2 = weight_variable([n_hidden_1, n_hidden_2])
    b2 = bias_variable([n_hidden_2])
    # Hidden Layer -> Output
    w3 = weight_variable([n_hidden_2, 1])
    b3 = bias_variable([1])

    # 1st Hidden layer with dropout
    h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
    h1_dropout = tf.nn.dropout(h1, keep_prob)
    # 2nd Hidden layer with dropout
    h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
    h2_dropout = tf.nn.dropout(h2, keep_prob)

    # Run sigmoid on output to get 0 to 1
    out = tf.nn.sigmoid(tf.matmul(h2_dropout, w3) + b3)

    # Loss function with L2 Regularization
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

    scaled_out = tf.multiply(out, MAX_Y)  # Scale output
    return inputs, out, scaled_out, regularizers

def main(_):
    tf.set_random_seed(RANDOM_SEED)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IN_DIM])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Dropout on hidden layers
    keep_prob = tf.placeholder("float")

    # Build the graph for the deep net
    inputs, out, scaled_out, regularizers = deep_network(x, keep_prob)

    # Normal loss function (RMSE)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, scaled_out))))

    # Loss function with L2 Regularization
    loss = tf.reduce_mean(loss + BETA * regularizers)

    # Optimizer
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    total_error = tf.reduce_sum(tf.square(tf.subtract(y_, tf.reduce_mean(y_))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_, scaled_out)))
    accuracy = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

    # Save model
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        #if RESTORE:
        #    print('Loading Model...')
        #    ckpt = tf.train.get_checkpoint_state('./models/neural/')
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        #else:
        sess.run(tf.global_variables_initializer())

        train, y_train, test_submit, test_submit_id = get_data()

        # Train until maximum steps reached or interrupted
        for i in range(START, STEPS):
            k_fold = KFold(n_splits=10, shuffle=True)
            #if i % 100 == 0:
            #    saver.save(sess, './models/neural/step_' + str(i) + '.cptk')

            for k, (ktrain, ktest) in enumerate(k_fold.split(train, y_train)):
                train_step.run(feed_dict={x: train[ktrain], y_: y_train[ktrain], keep_prob: DROPOUT})
                # Show test score every 10 iterations
                if i % 10 == 0:
                    # Tensorflow R2
                    #train_accuracy = accuracy.eval(feed_dict={
                    #    x: train[ktest], y_: y_train[ktest]})
                    # SkLearn metrics R2
                    train_accuracy = r2_score(y_train[ktest],
                                              sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
                    print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, train_accuracy))

        ####################
        # CV (repeat 5 times)
        ####################
        CV = []
        for i in range(5):
            k_fold = KFold(n_splits=10, shuffle=True)
            for k, (ktrain, ktest) in enumerate(k_fold.split(train, y_train)):
                # Tensorflow R2
                #accuracy = accuracy.eval(feed_dict={
                #    x: train[ktest], y_: y_train[ktest]})
                # SkLearn metrics R2
                accuracy = r2_score(y_train[ktest],
                                          sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
                print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, accuracy))
                CV.append(accuracy)
        print('Mean R2: %g' % (np.mean(CV)))

if __name__ == '__main__':
    tf.app.run()

