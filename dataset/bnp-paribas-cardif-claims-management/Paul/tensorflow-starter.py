import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

LEARNING_RATE = 1e-2
BATCH_SIZE = 256

train_path = '../input/train.csv'
test_path = '../input/test.csv'

def dense_to_one_hot(labels_dense, num_classes=10):
  """
  Convert class labels from scalars to one-hot vectors.
  http://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
  """
  labels_dense = np.array(labels_dense)
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def process_data(path, is_test_set=False):

    df = pd.read_csv(path)
    
    if is_test_set:
        df = df.interpolate()

    # Ignoring categorical input
    df = df.drop('v3', axis=1)
    df = df.drop('v22', axis=1)
    df = df.drop('v24', axis=1)
    df = df.drop('v30', axis=1)
    df = df.drop('v31', axis=1)
    df = df.drop('v47', axis=1)
    df = df.drop('v52', axis=1)
    df = df.drop('v56', axis=1)
    df = df.drop('v66', axis=1)
    df = df.drop('v71', axis=1)
    df = df.drop('v74', axis=1)
    df = df.drop('v75', axis=1)
    df = df.drop('v79', axis=1)
    df = df.drop('v91', axis=1)
    df = df.drop('v107', axis=1)
    df = df.drop('v110', axis=1)
    df = df.drop('v112', axis=1)
    df = df.drop('v113', axis=1)
    df = df.drop('v125', axis=1)
    
    if not is_test_set:
        # Ignore datapoints with missing values
        df = df.dropna()
        
        labels = df['target'].values.tolist()
        labels = dense_to_one_hot(labels, 2)
        df = df.drop('ID', axis=1)
        df = df.drop('target', axis=1)
        
        # Normalize the data
        df_norm = (df - df.mean()) / (df.max() - df.min())
        data = df_norm.values.tolist()

        X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                test_size=0.15, random_state=42)
        return X_train, X_test, y_train, y_test

    else:
        ids = df['ID'].values.tolist()
        df = df.drop('ID', axis=1)
        df_norm = (df - df.mean()) / (df.max() - df.min())
        data = df_norm.values.tolist()
        return data, ids


# Input nodes for passing data into the graph
x = tf.placeholder(tf.float32, [None, 112])
y = tf.placeholder(tf.float32, [None, 2])

# This is our model, a very simple, 1-layer MLP
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0]))
    return tf.matmul(layer_1, _weights[1]) + _biases[1]

weights = [
    tf.Variable(tf.random_normal([112, 50], seed=888)),
    tf.Variable(tf.random_normal([50, 2], seed=888))
]
biases = [
    tf.Variable(tf.random_normal([50], seed=888)),
    tf.Variable(tf.random_normal([2], seed=888))
]

pred = multilayer_perceptron(x, weights, biases)

# This is only used during test time
logits = tf.nn.softmax(pred)

# Run cross entropy with an Adam optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

with tf.Session() as sess:

    # We need to initialize the graph before we can use it
    tf.initialize_all_variables().run()
    
    X_train, X_test, y_train, y_test = process_data(train_path)
    data_len = len(X_train)
    

    print('\nTrain loss : Valid loss')

    total_t_loss = []
    for step in range(401):
        batch_index = (step*BATCH_SIZE)%(data_len+1-BATCH_SIZE)
        batch_data = X_train[batch_index:batch_index+BATCH_SIZE]
        batch_labels = y_train[batch_index:batch_index+BATCH_SIZE]

        _, l = sess.run([optimizer, loss], feed_dict={x: batch_data, y: batch_labels})
        total_t_loss.append(l)

        if step%100 == 0:
            avg_train_loss = np.mean(total_t_loss)
            total_v_loss = []
            for v_step in range(len(X_test)//BATCH_SIZE):
                batch_index = (v_step*BATCH_SIZE)%(len(X_test)+1-BATCH_SIZE)

                batch_data_v = X_test[batch_index:batch_index+BATCH_SIZE]
                
                batch_labels_v = y_test[batch_index:batch_index+BATCH_SIZE]

                l_v = sess.run([loss], feed_dict={x: batch_data_v, y: batch_labels_v})
            
                total_v_loss.append(l_v[0])

            print('{0} : {1}'.format(l, np.mean(total_v_loss)))
            total_t_loss = []

    # Build the submission file
    X_eval, ids = process_data(test_path, True) # '/Users/Peace/Projects/Cardif/data/raw/test.csv'

    outputs = sess.run([logits], feed_dict={x: X_eval})
    outputs = [x[1] for x in outputs[0]]

    submission = ['ID,PredictedProb']

    for prediction, id in zip(outputs, ids):
        submission.append('{0},{1}'.format(id,prediction))

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)



