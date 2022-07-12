""" Simple densely connected neural network to predict output, also contains preprocessing of the dataset
    such as encoding the categoricals into one_hot vectors and crossing out features which were constant.
    K-folds cross validation has also been implemented
    """

import pandas as pd
import numpy as np
import tensorflow as tf

#Constants
PATH = '../input/'
TRAIN = 'train.csv'
TEST = 'test.csv'
SUBMIT = False
LOGGING = 100

# Hyper Parameters
MAX_ITER = 1200
LEARNING_RATE = 1e-3
LAYERS = [100, 100]
FOLDS = 5
DROPOUT = 0.5

FEATURE_DROP = ['X11', 'X93', 'X107', 'X223', 'X235', 'X268', 'X289', 
'X290', 'X293', 'X297', 'X330', 'X347']


# Helper Functions
def one_hot(category, categories_dict):
    one_hot = np.zeros((1, len(categories_dict)), dtype='float32')
    idx = categories_dict[category[0]]
    one_hot.flat[idx] = 1
    
    return one_hot
    

# Helper class to perform K-Folds Validation splitting
class CrossValidationFolds(object):
    
    def __init__(self, data, labels, num_folds, shuffle=True):
        self.data = data
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0
        
        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.data.shape[0])
            data = data[perm]
            labels = labels[perm]
    
    def split(self):
        current = self.current_fold
        size = int(self.data.shape[0]/self.num_folds)
        
        index = np.arange(self.data.shape[0])
        lower_bound = index >= current*size
        upper_bound = index < (current + 1)*size
        cv_region = lower_bound*upper_bound

        cv_data = self.data[cv_region]
        train_data = self.data[~cv_region]
        
        cv_labels = self.labels[cv_region]
        train_labels = self.labels[~cv_region]
        
        self.current_fold += 1
        return (train_data, train_labels), (cv_data, cv_labels)
        
# Read Data
print('Reading CSV Data...')
train_df = pd.read_csv(PATH + TRAIN)
test_df = pd.read_csv(PATH + TEST)
num_examples = train_df.shape[0]  # Both the training and test set have the same # of examples
print('Data Read\n')

# ***Pre preocessing***
# Extracting targets
target = train_df['y'].values
target = target.reshape(num_examples, 1)
del train_df['y']

# Extracting ID Columns 
ID = test_df['ID'].values.reshape(num_examples)
del train_df['ID']
del test_df['ID']

# Delete constant features
for feature in FEATURE_DROP:
    del train_df[feature]
    del test_df[feature]
    # print('Dropped Feature {}'.format(feature))
    
# Categorical and binary features
categoricals = train_df.columns[train_df.dtypes == object]
binaries = train_df.columns[train_df.dtypes == 'int64']

# **Encode categoricals into one_hot vectors**
categoricals_train = np.empty((num_examples, 0))
categoricals_test = np.empty((num_examples, 0))

# Done in a feature by feature basis
for feature in categoricals:
    union = pd.Series(train_df[feature].tolist() +test_df[feature].tolist()).unique()
    union.sort()
    
    # Construct dict of categories in feaure
    feature_dict = {}
    for i in range(len(union)):
        feature_dict[union[i]] = i
        
    # Create one_hot accumulator
    train_one_hot = np.empty((0, len(union)))
    test_one_hot = np.empty((0, len(union)))
    
    # Create one_hot for each feature separetely, not a vectorized implementation and somewhat obscure
    for i in range(train_df.shape[0]):
        train_one_hot = np.concatenate((train_one_hot, one_hot(train_df[feature].values,feature_dict)))
        test_one_hot = np.concatenate((test_one_hot, one_hot(test_df[feature].values,feature_dict)))

    # Concatenate one_hot of each features into one_hot of all categoricals
    categoricals_train = np.concatenate((categoricals_train, train_one_hot), axis=1)
    categoricals_test = np.concatenate((categoricals_test, test_one_hot), axis=1)

# concatenate one_hot categoricals and binaries into a full input dataset
train_data = np.concatenate((categoricals_train, train_df[binaries].values.astype('float32')), axis=1)
test_input = np.concatenate((categoricals_test, test_df[binaries].values.astype('float32')), axis=1)

# Now let's get Tensorflowy, we now build our network, it will be a fairly simple 2 hidden layers densely connected network
# k-fold cross validation is now implementes and we can play with hyperparameter tuning
device = "/gpu:0"
config = tf.ConfigProto(allow_soft_placement=True,  device_count = {'GPU': 1})
config.gpu_options.allow_growth=True

# K-fold Cross Validation
r_squared_log = []
mse_log = []
data = CrossValidationFolds(train_data, target, FOLDS)
with tf.device(device):
    with tf.Session(config=config) as sess:
        
        x = tf.placeholder(tf.float32, shape=[None, 567])
        y_ = tf.placeholder(tf.float32, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float32)
        
        # First Layer
        W1 = tf.Variable(tf.truncated_normal([567, LAYERS[0]], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[LAYERS[0]]))

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

        # Second Layer
        W2 = tf.Variable(tf.truncated_normal([LAYERS[0], LAYERS[1]], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[LAYERS[1]]))

        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        # Dropout
        h2_drop = tf.nn.dropout(h2, keep_prob)

        # Output Layer
        W3 = tf.Variable(tf.truncated_normal([LAYERS[1], 1], stddev=0.1))
        b3 = tf.Variable(tf.constant(0.1, shape=[1]))

        y_fc = tf.matmul(h2_drop,W3) + b3

        # Loss function and optimizer
        loss = tf.losses.mean_squared_error(labels=y_, predictions=y_fc)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
        for i in range(FOLDS):
            print('Current fold: {}\n'.format(data.current_fold + 1))
            (train_input, train_target), (cv_input, cv_target) = data.split()
            
            # Start Training
            sess.run(tf.global_variables_initializer())
            print('Starting Training...')
            for i in range(MAX_ITER):
            
                if i % LOGGING == 0:
                    mse = loss.eval(feed_dict = {x: train_input, y_: train_target, keep_prob: 1.0})
                    cv_mse = loss.eval(feed_dict = {x: cv_input, y_: cv_target, keep_prob: 1.0})
                    
                    print('Step {0}, Train MSE: {1: .2f} | CV MSE: {2: .2f}'.format(i, mse, cv_mse))
                train_step.run(feed_dict = {x: train_input, y_: train_target, keep_prob: DROPOUT})
            
            mse = loss.eval(feed_dict = {x: train_input, y_: train_target, keep_prob: 1.0})
            r_squared = 1 - mse/np.var(train_target)
            
            cv_mse = loss.eval(feed_dict = {x: cv_input, y_: cv_target, keep_prob: 1.0})
            cv_r_squared = 1 - cv_mse/np.var(cv_target)
            
            print('\nTraining Finished, Training MSE: {0: .2f} | R_squared: {1: .5f}'.format(mse, r_squared))
            print('                 Validation MSE: {0: .2f} | R_squared: {1: .5f}\n'.format(cv_mse, cv_r_squared))
            
            mse_log.append(cv_mse)
            r_squared_log.append(cv_r_squared)
            
            if SUBMIT:
                inference = y_fc.eval(feed_dict = {x: test_input, keep_prob: 1.0})
                inference = inference.reshape(num_examples)
        
        sess.close()
        
        final_mse = np.array(mse_log).mean()
        final_r_squared = np.array(r_squared_log).mean()
        
        print('K folds finished')
        print('Final validation score, MSE: {0: .2f} | R_squared: {1: .5f}'.format(final_mse, final_r_squared))

if SUBMIT:
    predictions = pd.DataFrame({'ID': ID, 'y': inference})
    predictions.to_csv("NN.csv", index=False, header=True)