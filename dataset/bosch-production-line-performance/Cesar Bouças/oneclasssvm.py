import gc
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

NROWS = 50000
CHUNKSIZE = 50000
DATA_DIR = '../input/'

cols = [['Id',
             'L3_S29_D3474', 
             'L3_S30_D3496', 'L3_S30_D3506',
             'L3_S30_D3501', 'L3_S30_D3516',
             'L3_S30_D3511', 
             'L3_S32_D3852', 
             'L3_S33_D3856', 'L3_S33_D3858',
             'L3_S37_D3942',
             'L3_S37_D3943','L3_S37_D3945',
             'L3_S37_D3947','L3_S37_D3949',
             'L3_S37_D3951','L3_S38_D3953',
             'L3_S38_D3957','L3_S38_D3961'
             ],
            ['Id',
             'L0_S4_F109', 'L0_S15_F403', 'L0_S13_F354',
             'L1_S24_F1846', 'L1_S24_F1695', 'L1_S24_F1632', 'L1_S24_F1604',
             'L1_S24_F1723', 'L1_S24_F1844', 'L1_S24_F1842',
             'L2_S26_F3106', 'L2_S26_F3036', 'L2_S26_F3113', 'L2_S26_F3073',
             'L3_S29_F3407', 'L3_S29_F3376', 'L3_S29_F3324', 'L3_S29_F3382', 'L3_S29_F3479',
             'L3_S30_F3704', 'L3_S30_F3774', 'L3_S30_F3554',
             'L3_S32_F3850', 'L3_S32_F3850',
             'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3865',
             'L3_S37_F3944', 'L3_S37_F3946', 'L3_S37_F3948', 'L3_S37_F3950', 
             'L3_S38_F3956', 'L3_S38_F3960', 'L3_S38_F3952',
             'L3_S30_F3604', 'L3_S30_F3749', 'L0_S0_F20', 'L3_S30_F3559', 'L3_S30_F3819', 'L3_S29_F3321', 'L3_S29_F3373',
             'L3_S30_F3569', 'L3_S30_F3569', 'L3_S30_F3579', 'L3_S30_F3639', 'L3_S29_F3449', 'L3_S36_F3918', 'L3_S30_F3609',
             'L3_S30_F3574', 'L3_S29_F3354', 'L3_S30_F3759', 'L0_S6_F122', 'L3_S30_F3664', 'L3_S30_F3534', 'L0_S1_F24', 'L3_S29_F3342',
             'L0_S7_F138', 'L2_S26_F3121', 'L3_S30_F3744', 'L3_S30_F3799', 'L3_S33_F3859', 'L3_S30_F3784', 'L3_S30_F3769', 'L2_S26_F3040',
             'L3_S30_F3804', 'L0_S5_F114', 'L0_S12_F336', 'L0_S9_F170', 'L3_S29_F3330', 'L3_S29_F3351', 'L3_S29_F3339', 'L3_S29_F3427', 'L3_S30_F3829',
             'L0_S0_F22', 'L3_S30_F3589', 'L3_S30_F3494', 'L3_S29_F3421', 'L3_S29_F3327', 'L0_S5_F116', 'L3_S29_F3318', 'L3_S30_F3524', 'L3_S29_F3379',
             'L3_S29_F3333', 'L3_S29_F3455', 'L3_S29_F3430', 'L3_S30_F3529', 'L0_S0_F0', 'L3_S30_F3754', 'L3_S36_F3920', 'L0_S3_F96', 'L3_S29_F3407', 
             'L3_S29_F3473', 'L3_S29_F3476', 'L3_S30_F3674',
             'Response']]

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)
        
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc

def loadSet2(setname='test'):
    X = np.concatenate([
    pd.read_csv("../input/{0}_date.csv".format(setname), index_col=0, dtype=np.float32,
                usecols=cols[0], nrows=NROWS).values,
    pd.read_csv("../input/{0}_numeric.csv".format(setname), index_col=0, dtype=np.float32,
                usecols=cols[1][0:-1], nrows=NROWS).values
    ], axis=1)
    #if setname == 'train':
        #y = pd.read_csv("../input/{0}_numeric.csv".format(setname), nrows=NROWS, index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()
    #else:
     #   y = None
    #return X, y
    return X
    
def loadTrainSet(setname='train'):
    
    t = pd.merge(
    pd.read_csv("../input/{0}_date.csv".format(setname), index_col=0, dtype=np.float32,
                usecols=cols[0], nrows=NROWS),
    pd.read_csv("../input/{0}_numeric.csv".format(setname), index_col=0, dtype=np.float32,
                usecols=cols[1], nrows=NROWS), 
                how='right', left_index=True, right_index=True)
    
    return t

def loadTestSet(setname='test'):
    
    t = pd.merge(
    pd.read_csv("../input/{0}_date.csv".format(setname), index_col=0, dtype=np.float32,
                usecols=cols[0], nrows=NROWS),
    pd.read_csv("../input/{0}_numeric.csv".format(setname), index_col=0, dtype=np.float32,
                usecols=cols[1][0:-1], nrows=NROWS), 
                how='right', left_index=True, right_index=True)
    
    return t
    
def resultsToCsv(filename, ids, y):
    df = pd.DataFrame({'Id': ids, 'Response': y})
    count1 = df[df.Response == 1].shape[0]
    print('Response 1: {0} in {1}.csv'.format(count1, filename))
    if count1 > 0:
        df[['Id', 'Response']].to_csv("{0}.csv".format(filename), index=False)

# Convert an array of (1, -1) values to (0,1) values
def formatBin(y, inverted=False):
    # Inliers are labeled 1, while outliers are labeled -1
    # http://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection
    if inverted:
        return ((y.astype(int) + 1)//2)
    return ((y.astype(int) - 1)//-2)

def Train():
    train = loadTrainSet()
    train.fillna(train.mean(), inplace = True)
    print('Train:', train.shape)

    features = train.columns[1:-1]
    print(features)

    # Take data with Label = 0 to train
    X = preprocessing.scale(train[train.Response==0][features].values)
    #X = train[train.Response==0][features].values
    X_train, X_validate = train_test_split(X, test_size=0.25)

    # Take Label = 1 as anomalous data
    X_outliers = train[train.Response==1][features].values
    y_outliers = np.ones(X_outliers.shape[0]).astype(int)

    # Will pass dataset balance as parameter to the model
#    p_nu = X_outliers.shape[0] / X_train.shape[0]
    p_nu = X_outliers.shape[0] / train.shape[0]
    #p_nu = 0.005
    print('NU: {0}'.format(p_nu))

    # Debug
#    print(X_train.shape)
#    print(X_test.shape)
    #print('Full bin count: {0}'.format(np.bincount(train['Response'].astype(int))))
    print('Validation Outliers: {0}'.format(np.bincount(y_outliers)))
    print('Trainning Inliers:  {0}'.format(X_train.shape[0]))

    # Fill NaNs with the means
    #imp = Imputer(missing_values=NA, strategy='mean', axis=0).fit(train[features])
    # Normalize, maxabs_scale, scale
    #preprocessing.scale(imp.transform(X_mix))
    #imp.transform(X_train)
    #preprocessing.scale(imp.transform(X_test))

    # Free memory
    del train
    gc.collect()

    # Train
    #http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
    #http://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection
    print('OneClassSVM train:')
    clf = svm.OneClassSVM(nu=p_nu, 
                        kernel='rbf', 
                        gamma='auto').fit(X_train)


    # Validation
    print('OneClassSVM validation:')
    X_mix = np.concatenate((X_validate, X_outliers), axis=0)
    y_mix = np.concatenate((np.zeros(X_validate.shape[0]), np.ones(X_outliers.shape[0])), axis=0).astype(int)
    raw_y_mix_predicted = clf.predict(X_mix)
    #print('Validation set bin count:  {0}'.format(np.bincount(y_mix)))
    print(raw_y_mix_predicted)

    # Convert response values from -1/1 to 0/1
    y_mix_predicted = formatBin(raw_y_mix_predicted, inverted=False)
    
    # Report
    print('Expected:  {0}'.format(np.bincount(y_mix)))
    print(y_mix)
    print('Predicted: {0}'.format(np.bincount(y_mix_predicted)))
    print(y_mix_predicted)
    score = eval_mcc(y_mix, y_mix_predicted, show=False)
    print('Score: {0}'.format(score))
    print(score)
    
    #print('Inverted score:')
    #y_mix_predicted_inv = formatBin(raw_y_mix_predicted, inverted=True)
    #iscore= eval_mcc(y_mix, y_mix_predicted_inv, show=False)
    #print('Inv score: {0}'.format(iscore))

    # Load test data
    test = loadTestSet()
    test.fillna(test.mean(), inplace=True)
    print('Test', test.shape)
    X_test = preprocessing.scale(test[features].values)
    #X_test = test[features].values

    # Test
    y_pred = clf.predict(X_test)
    response = formatBin(y_pred)
    print(response)
    print(np.bincount(response))

    # Save the results and the trained model
    if NROWS > 10000000000:
        joblib.dump(clf, 'model_OneClassSVM.pkl')
        resultsToCsv('SVMsubmission', test.Id.values, response)


if __name__ == "__main__":
    print('Started')    
    #extractFeatures()
    #genetareCategoricalInt()
    Train()
    print('Finished')
    