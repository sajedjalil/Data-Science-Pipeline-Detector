__author__ = 'gs'
"""

Simplified version (without evaluating)
--------------------------------------

original author: Ben Hammer.

modifications: Harshaneel Gokhale
(LB - 0.985327 ensemble of UGBC+XGB+RF)

final modifications: Grzegorz Sionkowski
(LB - 0.991099 just UGBC) <- full version
Python ver. 2.7

Use as the basis for general testing of the hep_ml api: Mike Hall
https://arogozhnikov.github.io/hep_ml/index.html
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d
from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction
from hep_ml.losses import KnnAdaLossFunction
from hep_ml.losses import KnnFlatnessLossFunction
from hep_ml.nnet import MLPClassifier
from hep_ml import metrics

# -------------- switches ------------------------ #
DO_IMP = False
DO_BIN = False
DO_ADA = False
DO_KNNF = False
# -------------- loading data files -------------- #
print("Load the train/test/eval data using pandas")
train = pd.read_csv("../input/training.csv")
train = train[train['min_ANNmuon'] > 0.4]
test  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv('../input/check_agreement.csv', index_col='id')
corr_check = pd.read_csv("../input/check_correlation.csv")
signal = train.signal
trainids = train.index.values

#--------------- metrics ------------------ #
# maybe these could be tuned to better match up training and leaderboard results?
# these have seemed off to me sometimes
def check_correlation(probabilities, mass):
    probabilities, mass = map(column_or_1d, [probabilities, mass])

    y_pred = np.zeros(shape=(len(probabilities), 2))
    y_pred[:, 1] = probabilities
    y_pred[:, 0] = 1 - probabilities
    y_true = [0] * len(probabilities)
    df_mass = pd.DataFrame({'mass': mass})
    cvm = metrics.BinBasedCvM(uniform_features=['mass'], uniform_label=0, n_bins=15) # mjh overrode n_bins number
    cvm.fit(df_mass, y_true)
    return cvm(y_true, y_pred, sample_weight=None)

# knn based version of CvM
def check_correlation_knn(probabilities, mass):
    probabilities, mass = map(column_or_1d, [probabilities, mass])

    y_pred = np.zeros(shape=(len(probabilities), 2))
    y_pred[:, 1] = probabilities
    y_pred[:, 0] = 1 - probabilities
    y_true = [0] * len(probabilities)
    df_mass = pd.DataFrame({'mass': mass})
    cvm = metrics.KnnBasedCvM(uniform_features=['mass'], uniform_label=0, n_neighbours=50)
    cvm.fit(df_mass, y_true)
    return cvm(y_true, y_pred, sample_weight=None)

def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.

    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert np.all(predictions >= 0.) and np.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = np.minimum(tpr, tpr_thresholds[index])
        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))
    tpr_thresholds = np.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= np.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * np.array(roc_weights))
    return area   
    
def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve
    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = np.concatenate([sample_weights_zero, sample_weights_one])
    data_all = np.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr
    
def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.
    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = np.array(data_prediction), np.array(mc_prediction)
    weights_data, weights_mc = np.array(weights_data), np.array(weights_mc)

    assert np.all(data_prediction >= 0.) and np.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert np.all(mc_prediction >= 0.) and np.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= np.sum(weights_data)
    weights_mc /= np.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = np.max(np.abs(fpr - tpr))
    return Dnm
#--------------- metric importances ----------------#
def mc_metric(model, corr_check, features):
    y_corr = model.predict_proba(corr_check[features])[:, 1]
    corr_metric = check_correlation(y_corr, corr_check['mass'])
    return corr_metric
    
def mc_importances(model, corr_check, features):
    base_mc = mc_metric(model, corr_check, features)
    print("MC metric:",base_mc)
    imp_mc = []
    for col in features:
        save = corr_check[col].copy()
        corr_check[col] = np.random.permutation(corr_check[col])
        mc = mc_metric(model, corr_check, features)
        corr_check[col] = save
        imp_mc.append(mc - base_mc)
    return imp_mc
    
def auc_metric(model, X_train, y_train, features):
    y_pred = model.predict_proba(X_train[features])[:, 1]
    roc_auc = roc_auc_truncated(y_train, y_pred)
    return roc_auc
    
def auc_importances(model, X_train, y_train, features):
    base_auc = auc_metric(model, X_train, y_train, features)
    print("Baseline = ",base_auc)
    imp_auc = []
    for col in features:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m_auc = auc_metric(model, X_train, y_train, features)
        X_train[col] = save
        imp_auc.append(base_auc - m_auc)
    return imp_auc
#--------------- feature engineering -------------- #
def add_features(df):
    # features used by the others on Kaggle
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError'] # modified to:
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    # features from phunter
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # My:
    # new combined features just to minimize their number;
    # their physical sense doesn't matter
    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']
    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']
    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']
    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']
    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']
    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']
    #My:
    # "super" feature changing the result from 0.988641 to 0.991099
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df

print("Add features")
train = add_features(train)
test = add_features(test)
corr_check = add_features(corr_check)
check_agreement = add_features(check_agreement)

print("Eliminate features")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'SPDhits','CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',
              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',
              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',
              'p0_IP', 'p1_IP', 'p2_IP',
              'IP_p0p2', 'IP_p1p2',
              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',
              'DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in train.columns if f not in filter_out)

# mjh filter on metric importances, improve auc or reduce mass correlation
filter_imp = ['NEW_iso_def','NEW_iso_abc','NEW_pN_IP','iso_min','NEW_IP_pNpN',
              'FlightDistance','flight_dist_sig']

#features = list(f for f in features if f not in filter_imp)

# get validation data
train_val, test_val, y_train, y_test, train_id, test_id = train_test_split(train, signal, trainids, random_state=100, test_size=0.15, shuffle=True)

#-------------------  UGBC model -------------------- #
if DO_BIN:
    print("Train a UGradientBoostingClassifier")
    loss_bin = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0 , fl_coefficient=15, power=2)

    ugbc_bin = UGradientBoostingClassifier(loss=loss_bin, n_estimators=600,
                                 max_depth=6,
                                 learning_rate=0.15,
                                 train_features=features,
                                 subsample=0.7,
                                 random_state=123)
                                 
    ugbc_bin.fit(train_val[features + ['mass']], train_val['signal'])
    y_pred = ugbc_bin.predict_proba(test_val[features])[:, 1]
    roc_auc_bin = roc_auc_score(y_test, y_pred) 
    trunc_bin = roc_auc_truncated(y_test, y_pred)
    print("bin: auc:",roc_auc_bin,"auc trunc:",trunc_bin)
    y_mc = ugbc_bin.predict_proba(corr_check[features])[:, 1]
    mc = check_correlation(y_mc, corr_check['mass'])
    mc_knn = check_correlation_knn(y_mc, corr_check['mass'])
    print("bin: mass corr:",mc,"knn",mc_knn)
    agreement_probs = ugbc_bin.predict_proba(check_agreement[features])[:, 1]
    ks_bin = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print("bin: ks:",ks_bin)
# ------
# Determine importances
# ------
    if DO_IMP:
        imp_mc = mc_importances(ugbc_bin, corr_check, features)
        imp_auc = auc_importances(ugbc_bin, train, train['signal'], features)
        imp_df=pd.DataFrame(data=np.transpose(np.array([imp_auc,imp_mc])),index=features).sort_values(0,ascending=False)
        print(imp_df.tail(10))
# Train using all the data
    ugbc_bin.fit(train[features+['mass']], train['signal'])
    test_probs = ugbc_bin.predict_proba(test[features])[:,1]
    submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    submission.to_csv("ugbc_bin.csv", index=False)
    print("UGBC  BinFlatnessLossFunction Predictions done...")
    print("....")
#---------------
if DO_ADA:
    loss_ada = KnnAdaLossFunction(['mass'], uniform_label=0, knn=8)
    
    ugbc_ada = UGradientBoostingClassifier(loss=loss_ada, n_estimators=550,
                                 max_depth=5,
                                 learning_rate=0.13,
                                 train_features=features,
                                 subsample=0.8,
                                 random_state=123)
                                 
    ugbc_ada.fit(train_val[features + ['mass']], train_val['signal'])
    y_pred = ugbc_ada.predict_proba(test_val[features])[:, 1]
    roc_auc_ada = roc_auc_score(y_test, y_pred) 
    trunc_ada = roc_auc_truncated(y_test, y_pred)
    print("ada: auc:",roc_auc_ada,"auc trunc:",trunc_ada)
    y_mc = ugbc_ada.predict_proba(corr_check[features])[:, 1]
    mc = check_correlation(y_mc, corr_check['mass'])
    mc_knn = check_correlation_knn(y_mc, corr_check['mass'])
    print("ada: mass corr:",mc,"knn",mc_knn)
    agreement_probs = ugbc_ada.predict_proba(check_agreement[features])[:, 1]
    ks_ada = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print("ada: ks:",ks_ada)
    # Train using all the data
    ugbc_ada.fit(train[features+['mass']], train['signal'])
    test_probs = ugbc_ada.predict_proba(test[features])[:,1]
    submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    submission.to_csv("ugbc_ada.csv", index=False)
    print("UGBC  KnnAdaLossFunction Predictions done...")
    print("....")
#-------------------
if DO_KNNF:
    loss_knnf = KnnFlatnessLossFunction(['mass'], uniform_label=0, n_neighbours=24, fl_coefficient=15, power=2)

    ugbc_knnf = UGradientBoostingClassifier(loss=loss_knnf, n_estimators=550,
                                 max_depth=5,
                                 learning_rate=0.13,
                                 train_features=features,
                                 subsample=0.7,
                                 random_state=123)
                                 
    ugbc_knnf.fit(train_val[features + ['mass']], train_val['signal'])
    y_pred = ugbc_knnf.predict_proba(test_val[features])[:, 1]
    roc_auc_knnf = roc_auc_score(y_test, y_pred) 
    trunc_knnf = roc_auc_truncated(y_test, y_pred)
    print("knnf: auc:",roc_auc_knnf,"auc trunc:",trunc_knnf)
    y_mc = ugbc_knnf.predict_proba(corr_check[features])[:, 1]
    mc = check_correlation(y_mc, corr_check['mass'])
    mc_knn = check_correlation_knn(y_mc, corr_check['mass'])
    print("knnf: mass corr:",mc,"knn",mc_knn)
    agreement_probs = ugbc_knnf.predict_proba(check_agreement[features])[:, 1]
    ks_knnf = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print("knnf: ks:",ks_knnf)
# Train using all the data
    ugbc_knnf.fit(train[features+['mass']], train['signal'])
    test_probs = ugbc_knnf.predict_proba(test[features])[:,1]
    submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    submission.to_csv("ugbc_knnf.csv", index=False)
    print("UGBC KnnFlatnessLossFunction Predictions done...")
    print("....")
#-------------------
# Have a go at a neural net
clf = MLPClassifier(layers=[12,12], epochs=600)
#clf.fit(train_val[features + ['mass']], train_val['signal'])
clf.fit(train_val[features], train_val['signal'])
y_pred = clf.predict_proba(test_val[features])[:, 1]
roc_auc_nn = roc_auc_score(y_test, y_pred) 
trunc_nn = roc_auc_truncated(y_test, y_pred)
print("nn: auc:",roc_auc_nn,"auc trunc:",trunc_nn)
y_mc = clf.predict_proba(corr_check[features])[:, 1]
mc = check_correlation(y_mc, corr_check['mass'])
mc_knn = check_correlation_knn(y_mc, corr_check['mass'])
print("nn: mass corr:",mc,"knn",mc_knn)
agreement_probs = clf.predict_proba(check_agreement[features])[:, 1]
ks_nn = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print("nn: ks:",ks_nn)
# Train using all the data
clf.fit(train[features], train['signal'])
test_probs_nn = clf.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs_nn})
submission.to_csv("nn.csv", index=False)
print("UGBC MLPClassifier Predictions done...")
print("....")
#------ 
# adadelta trainer
# 10,10 500 adadelta showed very good accuracy bad mass corr
clf = MLPClassifier(layers=[15,15], epochs=500,trainer='adadelta')
clf.fit(train_val[features], train_val['signal'])
y_pred = clf.predict_proba(test_val[features])[:, 1]
roc_auc_nn = roc_auc_score(y_test, y_pred) 
trunc_nn = roc_auc_truncated(y_test, y_pred)
print("nn_ada: auc:",roc_auc_nn,"auc trunc:",trunc_nn)
y_mc = clf.predict_proba(corr_check[features])[:, 1]
mc = check_correlation(y_mc, corr_check['mass'])
mc_knn = check_correlation_knn(y_mc, corr_check['mass'])
print("nn_ada: mass corr:",mc,"knn",mc_knn)
agreement_probs = clf.predict_proba(check_agreement[features])[:, 1]
ks_nn = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print("nn_ada: ks:",ks_nn)
# Train using all the data
clf.fit(train[features], train['signal'])
test_probs_nn_ada = clf.predict_proba(test[features])[:,1]
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs_nn_ada})
submission.to_csv("nn_adadelta.csv", index=False)
print("UGBC MLPClassifier adadelta Predictions done...")
print("....")
#-------------------
# ensemble of nn's
# simple non-adadelta nn's seem to get decent accuracy with low mass correlation
# adadelta seem capable of very good accuracy with worse mass correlation
# try for a good accuracy low correlation ensemble of the two
p_weight = 0.90         
# Weighted average of the predictions:
result = pd.DataFrame({'id': test['id']})
result['prediction'] = 0.5*(p_weight*test_probs_nn + (1 - p_weight)*test_probs_nn_ada)
# Write to the submission file:
result.to_csv('nn_ens_sub.csv', index=False, header=["id", "prediction"], sep=',', mode='a')
#-------------------
# Have a go at an ensembled neural net
# 
#base_network = MLPClassifier(layers=[10], trainer='adadelta', trainer_parameters={'batch': 600})
if False:      # giving trainer_parameters errs out. this runs but slow, turning off for now.
    base_network = MLPClassifier(layers=[10], trainer='adadelta')
    abc = AdaBoostClassifier(base_estimator=base_network, n_estimators=20)
    abc.fit(train_val[features], train_val['signal'])
    y_pred = abc.predict_proba(test_val[features])[:, 1]
    roc_auc_nn = roc_auc_score(y_test, y_pred) 
    trunc_nn = roc_auc_truncated(y_test, y_pred)
    print("nn ada: auc:",roc_auc_nn,"auc trunc:",trunc_nn)
    y_mc = clf.predict_proba(corr_check[features])[:, 1]
    mc = check_correlation(y_mc, corr_check['mass'])
    mc_knn = check_correlation_knn(y_mc, corr_check['mass'])
    print("nn ada: mass corr:",mc,"knn",mc_knn)
    agreement_probs = clf.predict_proba(check_agreement[features])[:, 1]
    ks_nn = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print("nn ada: ks:",ks_nn)
    # Train using all the data
    clf.fit(train[features], train['signal'])
    test_probs = clf.predict_proba(test[features])[:,1]
    submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    submission.to_csv("nn.csv", index=False) 
    print("UGBC MLPClassifier Predictions done...")
    print("....")
#-------------------
# And importance functions for the addtional metrics
# not hep_ml specific but hep_ml still seems to have submission
# issues based on these metrics, so feature effects on the 
# metrics seems relevant.
#-------------------
#--------------------  prediction ---------------------#
#print ('----------------------------------------------')
#print("Make predictions on the test set")
#test_probs = ugbc.predict_proba(test[features])[:,1]
#submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
#submission.to_csv("justUGBC2.csv", index=False)