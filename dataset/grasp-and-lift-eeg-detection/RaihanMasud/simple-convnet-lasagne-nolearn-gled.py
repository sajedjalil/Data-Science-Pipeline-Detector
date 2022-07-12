
"""
@author: Raihan Masud in collaboration with Md Faijul Amin

Borrowing from Elena Cuoco's data loading... &
ConvNet Model from Denial Nouri's kfkd and Tim Hochberg's script

Todos for better accuracy:
    1. Filter data
    2. Use past history/sliding window data
    3. Do not use future data
    4. Down sample
    5. Increase batch_size inside BatchIterator
    6. Increase epoch size
    7. Tweaking layers, size, etc might improve accuracy
    8. Add accuracy calculation for better interpretation
"""

import numpy as np
import pandas as pd
from glob import glob
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano.tensor.nnet import sigmoid
from sklearn.preprocessing import StandardScaler
from lasagne.objectives import aggregate, binary_crossentropy

import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

#############function to read data###########
def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data', '_events')
    # read event file
    labels = pd.read_csv(events_fname)
    clean = data.drop(['id'], axis=1)  # remove id
    labels = labels.drop(['id'], axis=1)  # remove id
    return clean, labels


def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data


scaler = StandardScaler()


def data_preprocess_train(X):
    # normalize data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep


def data_preprocess_test(X):
    # normalizing data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep

#######columns name for labels#############
cols = ['HandStart', 'FirstDigitTouch',
        'BothStartLoadPhase', 'LiftOff',
        'Replace', 'BothReleased']

#######number of subjects###############
subjects = range(1, 13)
ids_tot = []
pred_tot = []
test_dict = dict()

def float32(k):
    return np.cast['float32'](k)


channels = 32
batch_size = None  #None = arbitary batch size
hidden_layer_size = 100  #change to 1024
N_EVENTS = 6
max_epochs = 8
NO_TIME_POINTS = 100

test_total = 0

def loss(x, t):
    return aggregate(binary_crossentropy(x, t))

######################## Deep Neural Network MODEL wih Convolution Layers######################

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv1', layers.Conv1DLayer),
        ('conv2', layers.Conv1DLayer),
        ('pool1', layers.MaxPool1DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden5', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape=(None, channels, NO_TIME_POINTS),
    dropout1_p=0.5,
    conv1_num_filters=4, conv1_filter_size=1,
    conv2_num_filters=8, conv2_filter_size=4, pool1_pool_size=4,
    dropout2_p=0.5, hidden4_num_units=hidden_layer_size,
    dropout3_p=0.5, hidden5_num_units=hidden_layer_size,
    dropout4_p=0.5, output_num_units=N_EVENTS, output_nonlinearity=sigmoid,

    batch_iterator_train = BatchIterator(batch_size=1000),
    batch_iterator_test = BatchIterator(batch_size=1000),

    y_tensor_type=theano.tensor.matrix,
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    objective_loss_function=loss,
    regression=True,

    max_epochs=max_epochs,
    verbose=1,
)
######################## Deep Neural Network MODEL wih Convolution Layers######################



###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw = []
    raw = []


    # ################ READ DATA ################################################
    fnames = glob('../input/train/subj%d_series*_data.csv' % (subject))

    for fname in fnames:
        data, labels = prepare_data_train(fname)
        raw.append(data)
        y_raw.append(labels)

    if raw and y_raw:
        X = pd.concat(raw)
        y = pd.concat(y_raw)

    # transform in numpy array
    # transform train data in numpy array
        X_train = np.asarray(X.astype(np.float32))
        y_train = np.asarray(y.astype(np.float32))

####process training data####
    X = X_train
    X = data_preprocess_train(X)
    total_time_points = len(X) // NO_TIME_POINTS

    no_rows = total_time_points * NO_TIME_POINTS
    X = X[0:no_rows, :]

    X = X.transpose()
    X_Samples = np.split(X, total_time_points, axis=1)
    X = np.asarray(X_Samples)

    y = y_train
    y = y[0:no_rows, :]
    y = y[::NO_TIME_POINTS, :]

    print("Training subject%d...." %(subject))
    net.fit(X,y)

################ Read test data #####################################
    
    fnames = glob('../input/test/subj%d_series*_data.csv' % (subject))

    test = []
    idx = []

    fnames.reverse()
    for fname in fnames:
        data = prepare_data_test(fname)
        test.append(data)
        idx.append(np.array(data['id']))

        data_size = len(data)
        series = 9 if 'series9' in fname else 10
        data_name = 'subj{0}_series{1}'.format(subject, series)
        test_dict[data_name] = data_size

        test_total += data_size

    if idx and test:
        X_test = pd.concat(test)
        ids = np.concatenate(idx)
        ids_tot.append(ids)
        X_test = X_test.drop(['id'], axis=1)  # remove id
    # transform test data in numpy array
    X_test = np.asarray(X_test.astype(np.float32))


####process test data####
    X_test = X_test
    X_test = data_preprocess_test(X_test)
    total_test_time_points = len(X_test) // NO_TIME_POINTS
    remainder_test_points = len(X_test) % NO_TIME_POINTS

    no_rows = total_test_time_points * NO_TIME_POINTS
    X_test = X_test[0:no_rows, :]

    X_test = X_test.transpose()
    X_test_Samples = np.split(X_test, total_test_time_points, axis=1)
    X_test = np.asarray(X_test_Samples)


###########################################################################
#######get predictions and write to files for series 9 and series 10#######
    print("Testing subject%d...." %(subject))
    params = net.get_all_params_values()
    learned_weights = net.load_params_from(params)
    probabilities = net.predict_proba(X_test)

    sub9 = 'subj{0}_series{1}'.format(subject, 9)
    data_len9 = test_dict[sub9]
    total_time_points9 = data_len9 // NO_TIME_POINTS
    remainder_data9 = data_len9 % NO_TIME_POINTS

    sub10 = 'subj{0}_series{1}'.format(subject, 10)
    data_len10 = test_dict[sub10]
    total_time_points10 = data_len10 // NO_TIME_POINTS
    remainder_data10 = data_len10 % NO_TIME_POINTS

    total_test_points = total_time_points9+total_time_points10

    for i, p in enumerate(probabilities):
        if i != total_test_points:
            for j in range(NO_TIME_POINTS):
                pred_tot.append(p)
        if i+1 == total_time_points9 :
            for k in range(remainder_data9):
                pred_tot.append(pred_tot[-1])
    for k in range(remainder_data10):
        pred_tot.append(pred_tot[-1])

# submission file
print('Creating submission(prediction) file...')
prediction_file = './gled_conv_net_grasp.csv'
submission_file = './submission_gled_conv_net_grasp.csv'

# # create pandas object for sbmission

predictionDf = pd.DataFrame(index=np.concatenate(ids_tot),
                           columns=cols,
                           data=pred_tot)
# # write file
predictionDf.to_csv(prediction_file, index_label='id', float_format='%.6f')

predictionDf.apply( lambda x: np.where(x == x.max() , 1 , 0) , axis = 1)
predictionDf.to_csv(submission_file, index_label='id')


# submission file
