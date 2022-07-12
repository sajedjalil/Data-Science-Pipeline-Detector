#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost
from sklearn import preprocessing

NBINS=2 # Randomly pick this number of bins and fit
# See line 92 for iterating over the whole dataset

def horizontally_bin_data(data, NX, NY):
    """Add columns to data indicating X and Y bins.

    Divides the grid into `NX` bins in X and `NY` bins in Y, and adds columns 
    to `data` containing the bin number in X and Y. 
    """

    NX = int(NX)
    NY = int(NY)

    assert((NX >= 5) and (NX <= 1000))
    assert((NY >= 5) and (NY <= 1000))

    x_bounds = (0., 10.)
    y_bounds = (0., 10.)

    delta_X = (x_bounds[1] - x_bounds[0]) / float(NX)
    delta_Y = (y_bounds[1] - y_bounds[0]) / float(NY)

    # very fast binning algorithm, just divide by delta and round down
    xbins = np.floor((data.x.values - x_bounds[0])
                     / delta_X).astype(np.int32)
    ybins = np.floor((data.y.values - y_bounds[0])
                     / delta_Y).astype(np.int32)

    # some points fall on the upper/right edge of the domain
    # tweak their index to bring them back in the box
    xbins[xbins == NX] = NX-1
    ybins[ybins == NY] = NY-1

    xlabel = 'x_bin_{0:03d}'.format(NX)
    ylabel = 'y_bin_{0:03d}'.format(NY)

    data[xlabel] = xbins
    data[ylabel] = ybins
    return


def mapkprecision(truthvalues, predictions):
    '''
    This is a faster implementation of MAP@k valid for numpy arrays.
    It is only valid when there is one single truth value. 

    m ~ number of observations
    k ~ MAP at k -- in this case k should equal 3

    truthvalues.shape = (m,) 
    predictions.shape = (m, k)
    '''
    z = (predictions == truthvalues[:, None]).astype(np.float32)
    weights = 1./(np.arange(predictions.shape[1], dtype=np.float32) + 1.)
    z = z * weights[None, :]
    return np.mean(np.sum(z, axis=1))


def CV_fit(train, test, kfold=5, keep_fraction=0.5):
    '''Performs a fit using XGBoost. Applies kfold cross validation to 
    estimate error. 

    Parameters
    ----------
    train 
        The training dataset as a pandas DataFrame
    test
        The testing dataset as a pandas DataFrame
    kfold
        The number of folds to use for K Fold Validation
    keep_fraction
        A float between 0 and 1. The fraction of events in each bin
        to keep while minimizing the number of place_ids in the training
        set. Low values throw away a lot of infrequent place_ids, and
        values near 1 retain almost all place_ids. 
    '''

    # choose 10 random bins for a quicker estimate of algorithm error.
    rs = np.random.RandomState(42)
    bin_numbers = zip(rs.randint(0, 50, size=NBINS), rs.randint(0, 50, size=NBINS))

    predictions = []
    map3s = []

    #Choose this line for the whole dataset.
    #for i_bin_x, i_bin_y in itertools.product(xrange(50), xrange(50)):
    for i_bin_x, i_bin_y in bin_numbers:
        print("Bin {},{}".format(i_bin_x, i_bin_y))

        # choose the correct bin, sort values in time to better simulate
        # the real train/test split for k-fold validation
        train_in_bin = train[(train.x_bin_050 == i_bin_x)
                             & (train.y_bin_050 == i_bin_y)].sort_values('time')
        test_in_bin = test[(test.x_bin_050 == i_bin_x)
                           & (test.y_bin_050 == i_bin_y)].sort_values('time')

        N_total_in_bin = train_in_bin.shape[0]
        keep_N = int(float(N_total_in_bin)*keep_fraction)
        vc = train_in_bin.place_id.value_counts()

        # eliminate all ids which are low enough frequency
        vc = vc[np.cumsum(vc.values) < keep_N]
        df1 = pd.DataFrame({'place_id': vc.index, 'freq': vc.values})

        # this represents the training set after all low frequency place_ids
        # are removed
        train_in_bin_2 = pd.merge(train_in_bin, df1, on='place_id',
                                  how='inner')

        # XG Boost requires labels from 0 to n_labels, not place_ids
        le = preprocessing.LabelEncoder()
        le.fit(train_in_bin_2.place_id.values)
        y_train = le.transform(train_in_bin_2.place_id.values)

        # select columns (features) and make a numpy array
        x_train = train_in_bin_2['x y accuracy hour'.split()].as_matrix()
        x_test = test_in_bin['x y accuracy hour'.split()].as_matrix()

        # Construct DMatrices
        dm_train = xgboost.DMatrix(x_train, label=y_train)
        dm_test = xgboost.DMatrix(x_test)

        # use the XGBoost built in cross validation function,
        # stopping early to prevent overfitting
        res = xgboost.cv(
            {'eta': 0.1, 'objective': 'multi:softprob',
             'num_class': len(le.classes_),
             'alpha': 0.1, 'lambda': 0.1, 'booster': 'gbtree'},
            dm_train, num_boost_round=200, nfold=kfold, seed=42,
            early_stopping_rounds=10, verbose_eval=10
            # For some reason, verbose_eval seems to be broken on my install
        )

        print(res)

        # this will be the number of epochs that (approximately) prevents
        # overfitting
        N_epochs = res.shape[0]

        # For some reason, verbose_eval seems to be broken on my install
        booster = xgboost.train(
            {'eta': 0.1, 'objective': 'multi:softprob',
             'num_class': len(le.classes_),
             'alpha': 0.1, 'lambda': 0.1, 'booster': 'gbtree'},
            dm_train, num_boost_round=N_epochs, verbose_eval=10)
        predict_y_train = booster.predict(dm_train)
        predict_y_test = booster.predict(dm_test)

        # There is a DELIBERATE error here where the CV MAP@3 is misleadingly
        # good as compared to the true MAP@3 on the test set. I leave it as
        # an exercise how to fix this.

        # A top k algorithm would be theoretically faster, but benchmarks
        # indicate that with Order(50) elements, an argsort is faster than
        # heapq.

        predicted_train_idx = np.argsort(
            predict_y_train, axis=1)[:, -3:][:, ::-1]
        predicted_test_idx = np.argsort(
            predict_y_test, axis=1)[:, -3:][:, ::-1]

        c = np.array(le.classes_)
        predicted_train_place_id = np.take(c, predicted_train_idx)
        predicted_test_place_id = np.take(c, predicted_test_idx)

        map3 = mapkprecision(c.take(y_train), predicted_train_place_id)
        map3s.append(map3)

        print("Train Cross Validated MAP@3: {0:.4f}".format(map3))
        result = pd.DataFrame({'row_id': test_in_bin.index,
                               'pred_1': predicted_test_place_id[:, 0],
                               'pred_2': predicted_test_place_id[:, 1],
                               'pred_3': predicted_test_place_id[:, 2]})
        predictions.append(result)

    return pd.concat(predictions), map3s


def main():

    print('Reading csv files from disk.')
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    print("Calculating time features")

    train['hour'] = (train['time']//60) % 24+1  # 1 to 24
    test['hour'] = (test['time']//60) % 24+1  # 1 to 24

    # You'll probably want to generate more features here.

    print("Binning data")
    horizontally_bin_data(train, 50, 50)
    horizontally_bin_data(test, 50, 50)

    print("Starting CV")
    predictions, map3s = CV_fit(train, test, kfold=5)
    print("MAP@3 CV in bins {}".format(map3s))

    print("Writing xgb_submission.csv")
    with open('xgb_submission.csv', 'w') as fh:
        fh.write('row_id,place_id\n')
        for r in predictions.itertuples():
            fh.write("{0},{1} {2} {3}\n".format(*r))

if __name__ == '__main__':
    main()
