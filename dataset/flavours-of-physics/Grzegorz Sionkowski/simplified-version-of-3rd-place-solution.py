# Simplified version of my competition solution. It is not functionally
# equivalent to the solution I used in the competition but it contains the
# main tricks. It scores 0.999594 on private leaderboard as a 
# "Post-Deadline Entry" (~ third place)
#
# Some parts of this code are taken from:
# https://www.kaggle.com/justfor/flavours-of-physics/gridsearchcv-with-feature-in-xgboost
# evaluation module is taken from
# https://github.com/yandexdataschool/flavours-of-physics-start
#
# WARNING: sometime I'm working with column vectors, sometime with row vectors

import sys;
import random;
import pandas as pd;
import numpy as np;
import xgboost as xgb;
from sklearn.metrics import roc_curve, auc;
# "sys.path.append('../input')" enables evaluation module taken from
# https://github.com/yandexdataschool/flavours-of-physics-start
sys.path.append('../input')
import evaluation;

##################### functions and classes declarations #######################

# another implementation of PCA
class PCA(object):
    mu = np.zeros(1);
    scale = np.zeros(1);
    scaleInv = np.zeros(1);
    base = np.zeros(1);

    # constructor
    # @param x data to init PCA from, column vectors
    def __init__(self, x):
        # mean and covariance
        mu = np.mean(x, axis=1);
        mu = np.reshape(mu, (mu.size, 1));
        self.mu = mu.copy();
        sigma = np.cov(x, bias = 1);
        # eigenvalue decomposition
        w, v = np.linalg.eig(np.linalg.inv(sigma));
        v = np.transpose((np.transpose(v))[(-np.abs(w)).argsort()]).copy();
        w = -np.sort(-w)
        self.base = v.copy();
        # scale matrices
        ds = np.subtract(x, self.mu);
        sigma2 = np.cov(np.dot(np.transpose(self.base), ds), bias = 1);
        sigmaDiag = np.diag(sigma2).copy();
        sigmaDiag[sigmaDiag < 1.0e-12] = 1.0e-12;
        scaleFlat = np.sqrt(sigmaDiag) ** -1;
        scale = np.multiply(np.eye(scaleFlat.size), np.transpose(scaleFlat));
        self.scale = scale.copy();
        self.scaleInv = np.linalg.inv(self.scale).copy();

    # converts given data to PCA base
    # @param xs data as column vectors
    # @param dim required dimensionality of output
    # @return data from xs transformed to PCA base and scaled to zero mu,
    # unit sigma (scaling algorithm is probably wrong imlemeted)
    def toPCABase(self, xs, dim):
        ds = np.subtract(xs, self.mu);
        ds = np.dot(np.transpose(self.base[:, 0:dim]), ds);
        return np.dot(self.scale[0:dim, 0:dim], ds);

# calculates score in terms of ROC
# @param pp prediction, nx1 matrix 
# @param yy target values, nx1 matrix  
# @return score in terms of ROC
def getErrROC(pp, yy):
    # sort pp, yy according to pp values
    idxs = np.argsort(pp, 0);
    lcp = np.zeros(pp.shape);
    lcy = np.zeros(pp.shape);
    for i in range(0, yy.size):
        lcp[i] = pp[idxs[i]]
        lcy[i] = yy[idxs[i]]
    p = lcp;
    y = lcy;
    # calculate tpr, fpr
    tpr = np.zeros(p.shape);
    fpr = np.zeros(p.shape);
    sumP = np.sum(1.0 * y);
    sumN = np.sum(1 - 1.0 * y);
    tp = sumP;
    fp = sumN;
    for i in range(0, y.size):
        if y[i] == 1:
            tp = tp - 1;
        elif y[i] == 0:
            fp = fp - 1;
        else:
            raise ValueError('unknown y: %d' % y[i]);
        tpr[i] = tp / sumP;
        fpr[i] = fp / sumN;
    tpr = np.flipud(tpr);
    fpr = np.flipud(fpr);
    # calculate weighted area under ROC
    area = 0;
    for i in range(0, y.size - 1):
        dx = 1 - 0.5 * (fpr[i] + fpr[i + 1]);
        dy = tpr[i + 1] - tpr[i];
        locy = tpr[i + 1];
        w = -1;
        if (locy >= 0.8):
            w = 0.0;
        elif (locy >= 0.6):    
            w = 0.5;
        elif (locy >= 0.4):    
            w = 1.0;
        elif (locy >= 0.2):    
            w = 1.5;
        else:
            w = 2.0;
        area = area + w * dx * dy;
    return area;

# calculate value of agreement test
# @param pp predictions, nx1 matrix
# @param yy target values, nx1 matrix
# @param ww weights of individual datapoints, nx1 matrix 
def getErrAGR(pp, yy, ww):   
    # sort pp, yy, ww according to pp values
    idxs = np.argsort(pp, 0);
    lcp = np.zeros(pp.shape);
    lcy = np.zeros(pp.shape);
    lcw = np.zeros(pp.shape);
    for i in range(0, yy.size):
        lcp[i] = pp[idxs[i]]
        lcy[i] = yy[idxs[i]]
        lcw[i] = ww[idxs[i]]
    p = lcp;
    y = lcy;
    w = lcw;
    # calculate weighted tpr, fpr
    tpr = np.zeros(p.shape);
    fpr = np.zeros(p.shape);
    sumP = np.sum(np.multiply(1.0 * y, w));
    sumN = np.sum(np.multiply(1 - 1.0 * y, w));
    tp = sumP;
    fp = sumN;
    for i in range(0, y.size):
        if y[i] == 1:
            tp = tp - w[i];
        elif y[i] == 0:
            fp = fp - w[i];
        else:
            raise ValueError('unknown y: %d' % y[i]);
        tpr[i] = tp / sumP;
        fpr[i] = fp / sumN;
    # return maximum distance    
    return np.max(np.abs(tpr - fpr));    

# auxiliary method, loads data from csv file, posiibly filtering out
# lines with min_ANNmuon < 0.4
# @param path path to csv file
# @param doFilter if set to true, values with min_ANNmuon < 0.4 are filtered
# out
# @return pandas data frame containig data from given file
def slLoadCsv(path, doFilter):    
    matrix = pd.read_csv(path);
    if (not doFilter):
        return matrix;
    else:    
        filterIdxs = matrix["min_ANNmuon"] >= 0.4
        tmp = matrix[filterIdxs];
        return pd.DataFrame(tmp.values, columns=matrix.columns);

# transforms pandas dataframe to PCA base (without reducing dimensionality)
# @param pca PCA instance
# @param dataFrame data to be transformed
# @return DataFrame containing transformed data
def toPCABase(pca, dataFrame):    
    x = np.transpose(dataFrame.values);
    x = pca.toPCABase(x, x.shape[0]);
    return pd.DataFrame(np.transpose(x).copy());

# converts data to input used in XGB (applies PCA, adds estimated mass)
# @param pca PCA instance
# @param dataFrame dataFrame loaded from input file with some columns eliminated
# and applied add_features()
def toXgbInput(pca, dataFrame):
    x = np.transpose(dataFrame.values);
    x = pca.toPCABase(x, x.shape[0]);
    x = np.transpose(x).copy();
    return pd.DataFrame(x.copy());   

# Calculates some usefull features, provided by
# https://www.kaggle.com/justfor/flavours-of-physics/gridsearchcv-with-feature-in-xgboost
def add_features(df):
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    return df   

############################# the program self #################################

print("Load the data using pandas");
agr_test = slLoadCsv("../input/check_agreement.csv", False);
check_correlation = slLoadCsv("../input/check_correlation.csv", False);
train = slLoadCsv("../input/training.csv", False);
test = slLoadCsv("../input/test.csv", False);
rows10 = random.sample(list(train.index), int(train.shape[0] / 10));
sl_cv = train.ix[rows10];
sl_train = train.drop(rows10);

# create feature list
features0 = list(sl_train.columns[1:-5])
tmp = add_features(sl_train[features0]);        
features = list(tmp.columns);
# add calculated features
sl_train = add_features(sl_train);
sl_cv = add_features(sl_cv);        
agr_test = add_features(agr_test);        
check_correlation = add_features(check_correlation);        
train = add_features(train);        
test = add_features(test);        

print("init PCA")
pca = PCA(np.transpose(sl_train[features].values));        

# declare some auxiliar variables
print("prepare inputs")
sl_trainInputStrong = xgb.DMatrix(toXgbInput(pca, sl_train[features]), sl_train["signal"]);
sl_cvInputStrong = xgb.DMatrix(toXgbInput(pca, sl_cv[features]));
agrInputStrong = xgb.DMatrix(toXgbInput(pca, agr_test[features]));
correlInputStrong = xgb.DMatrix(toXgbInput(pca, check_correlation[features]));
trainInputStrong = xgb.DMatrix(toXgbInput(pca, train[features]), train["signal"]);
testInputStrong = xgb.DMatrix(toXgbInput(pca, test[features]));
sl_trainInputWeak = xgb.DMatrix(sl_train[features], sl_train["signal"]);
sl_cvInputWeak = xgb.DMatrix(sl_cv[features]);
agrInputWeak = xgb.DMatrix(agr_test[features]);
correlInputWeak = xgb.DMatrix(check_correlation[features]);
trainInputWeak = xgb.DMatrix(train[features], train["signal"]);
testInputWeak = xgb.DMatrix(test[features]);
sl_cvTarget = sl_cv["signal"].values;
agrTarget = agr_test["signal"];
agrW = agr_test["weight"];

print("Train XGBoost models")
xgbParams = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "lambda": 1.0,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
numTreesStrong = 350;
numTreesWeak = 13;
gbmPcaStrong = xgb.train(xgbParams, sl_trainInputStrong, numTreesStrong);
gbmPcaWeak = xgb.train(xgbParams, sl_trainInputWeak, numTreesWeak);

print("Make predictions on CV, agreement, correlation sets")
sl_cvPStrong = gbmPcaStrong.predict(sl_cvInputStrong);
agrPStrong = gbmPcaStrong.predict(agrInputStrong);
correlPStrong = gbmPcaStrong.predict(correlInputStrong);
sl_cvPWeak = gbmPcaWeak.predict(sl_cvInputWeak);
agrPWeak = gbmPcaWeak.predict(agrInputWeak);
correlPWeak = gbmPcaWeak.predict(correlInputWeak);

print("Experiment with ensembling")
# I accidentaly found, that raising XGB prediction to some exponent helps.
# I don't know why :-)
# The higher exponent the better it seems to work. 
for exponent in [1, 16, 256]:
    print('\nPredictions for exponent = %d:' % (exponent));
    for q in np.arange(0.1, 1.01, 0.1):
        sl_cvP = q * (sl_cvPStrong ** exponent) + (1 - q) * sl_cvPWeak;
        sl_cvP[sl_cvP < 0] = 0;
        sl_cvP[sl_cvP > 1] = 1;
        agrP = q * (agrPStrong ** exponent) + (1 - q) * agrPWeak;
        correlP = q * (correlPStrong ** exponent) + (1 - q) * correlPWeak;
        sqErr = np.mean((sl_cvP - sl_cvTarget) ** 2);
        roc = getErrROC(sl_cvP, sl_cvTarget);
        agr = getErrAGR(agrP, agrTarget, agrW);
        correl = evaluation.compute_cvm(correlP, check_correlation['mass']);
        print("mix %4.2f - sqErr: %6.4f, ROC: %6.4f, AGR: %6.4f, correl: %6.4f" \
                % (q, sqErr, roc, agr, correl))

print("Train XGBoost models on full data");
gbmPcaStrong = xgb.train(xgbParams, trainInputStrong, numTreesStrong);
gbmPcaWeak = xgb.train(xgbParams, trainInputWeak, numTreesWeak);

print("Make predictions on full data");
testPStrong = gbmPcaStrong.predict(testInputStrong);
testPWeak = gbmPcaWeak.predict(testInputWeak);

# mix xgb and nn prediction on test data, save output
for q in np.arange(0.1,1.01,0.1):
    testP = q * (testPStrong ** 256) + (1 - q) * testPWeak;
    testP[testP < 0] = 0;
    testP[testP > 1] = 1;
    submission = pd.DataFrame({"id": test["id"], "prediction": testP})
    fname = ("rf_xgboost_submission.%4.2f.csv" % (q));
    submission.to_csv(fname, index=False);
    print("%s saved" % fname);
