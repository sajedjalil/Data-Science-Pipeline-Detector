import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, boxcar
from numpy import convolve
from sklearn.linear_model import LogisticRegression
from glob import glob
import os
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.qda import QDA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ---- functions ----------------------------------------------
def prepare_data_train(fname, subsample):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data', '_events')
    # read event file
    labels = pd.read_csv(events_fname)
    # remove id column
    clean = data.drop(['id' ], axis=1)
    labels = labels.drop(['id' ], axis=1)
    # sub sampling
    clean = clean.loc[::subsample, :]
    labels = labels.loc[::subsample, :]
    return clean, labels


def prepare_data_test(fname):
    """ read and prepare test data """
    data = pd.read_csv(fname)
    return data


def butterworth_filter(X,t,k,l):
    if t == 0:
        freq = [k, l]
        b, a = butter(3, np.array(freq)/500.0, btype='bandpass')
        X = lfilter(b, a, X)
    elif t == 1:
        b, a = butter(3,k/500.0, btype='lowpass')
        X = lfilter(b, a, X)
    elif t == 2:
        b, a = butter(3, l/500.0, btype='highpass')
        X = lfilter(b, a, X)
    return X


def data_preprocess(X):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    # do here your pre processing

    return X_prep

# end of the functions ---------------------------------------------------------------------------

# start of the program -----
scaler = StandardScaler()

# if you want to down sample the training data
subsample = 100

# columns name for labels
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']


# number of subjects
subjects = range(1, 13)  # change to 13 for the 12 subjects
series = range(1, 9)  # there are 8 series and start in 1 not 0
ids_tot = []
pred_tot = []
rng = np.random.RandomState(0)

# loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    y_raw = []
    raw = []

    # ---- Read train data and Labels ------------------------------------
    fnames = glob('../input/train/subj%d_series1_data.csv' % subject)
    for fname in fnames:
        data,labels = prepare_data_train(fname, subsample)
        raw.append(data)
        y_raw.append(labels)
    X = pd.concat(raw)
    y = pd.concat(y_raw)
    # ---------------------------------------------------------------------

    # transform train data in numpy array
    X_train = np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))

    # ---- Read test data -------------------------------------------------
    fnames = glob('../input/test/subj%d_series*_data.csv' % subject)
    test = []
    idx = []
    for fname in fnames:
        data = prepare_data_test(fname)
        test.append(data)
        idx.append(np.array(data['id']))
    X_test = pd.concat(test)
    # ---------------------------------------------------------------------

    # create id for submission file
    ids = np.concatenate(idx)
    ids_tot.append(ids)

    # remove id from test file
    X_test = X_test.drop(['id'], axis=1)

    # transform test data in numpy array
    X_test =np.asarray(X_test.astype(float))

    # Data pre process: scaling or normalization and butter-worth filter
    X_train = data_preprocess(X_train)
    X_test = data_preprocess(X_test)

    # create matrices for submission file (prediction) 
    prediction = np.empty((X_test.shape[0],6))

    # ---- Create the classifier object
    lr1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=50)

    # classification per subject and label, one by one
    for i in range(6):
        # selection of the corresponding Label
        y_train = y[:, i]
        print('Train subject %d, class %s' % (subject, cols[i]))

        # train with a sub sampled version of the data
        lr1.fit(X_train, y_train)

        # calculation of the output for the submission file and ROC validation
        prediction[:, i] = lr1.predict_proba(X_test)[:, 1]  # lr3.predict_proba[:,1]

    # put the prediction into a list array
    pred_tot.append(prediction)


# ---- submission file ---------------------------------------------------------------------------------------------
submission_file = 'Submission.csv'
# create pandas object for submission
submission = pd.DataFrame(index=np.concatenate(ids_tot), columns=cols, data=np.concatenate(pred_tot))
# write file
print("writing file...")
submission.to_csv(submission_file,index_label='id',float_format='%.3f')
# ------------------------------------------------------------------------------------------------------------------

