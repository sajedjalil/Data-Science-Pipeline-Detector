
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from hep_ml.gradientboosting import UGradientBoostingClassifier,LogLossFunction
import xgboost as xgb
from sklearn.metrics import roc_curve, auc


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = np.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = np.cumsum(np.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return np.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=150, step=40):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = np.array(predictions)
    masses = np.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[np.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = np.argsort(np.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return np.mean(cvms)


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



print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

#randomize the training sample
train=train.iloc[np.random.permutation(len(train))]



print("Eliminate SPDhits, which makes the agreement check fail")
features= ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP',
       'IPSig', 'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree',
       'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
       'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2',
       
       
       'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_IP',
       'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 
            'p0_eta', 'p1_eta',
       'p2_eta']

print("Train a Random Fores and gradient boos model model")

"""
rf = RandomForestClassifier(n_estimators=100,random_state=5)
rf.fit(train[features],train["signal"])


"""
params = {"objective": "binary:logistic",
          "eta": 0.16,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=100
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

"""
ada= AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100,random_state=5),
                        n_estimators=600, random_state=5,learning_rate=0.25)
ada.fit(train[features],train["signal"])
"""
print("train a UBoost classifier")
loss_funct=LogLossFunction()
ub=UGradientBoostingClassifier(loss=loss_funct,n_estimators=100, random_state=3,learning_rate=0.256,subsample=0.7)
ub.fit(train[features],train["signal"])

"""
print("train a Gradientboost classifier")
gb=GradientBoostingClassifier(n_estimators=500, random_state=3,learning_rate=0.256,subsample=0.7,max_features=34)
gb.fit(train[features],train["signal"]
"""

check_agreement = pd.read_csv('../input/check_agreement.csv', index_col='id')


agreement_probs = 0.2*ub.predict_proba(check_agreement[features])[:, 1]+0.8*gbm.predict(xgb.DMatrix(check_agreement[features])) 
ks =compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

print ("KS metric",ks, ks < 0.09)




check_correlation = pd.read_csv('../input/check_correlation.csv', index_col='id')


correlation_probs =0.2*ub.predict_proba(check_correlation[features])[:, 1]+0.8*gbm.predict(xgb.DMatrix(check_correlation[features]))

cvm = compute_cvm(correlation_probs, check_correlation['mass'])
print ("CvM metric",cvm, cvm < 0.002)



train_eval = train[train['min_ANNmuon'] > 0.4]
print("calculating train probs having min_annmuon>0.4")
train_probs = 0.2*ub.predict_proba(train_eval[features])[:, 1]+0.8*gbm.predict(xgb.DMatrix(train_eval[features])) 

AUC = roc_auc_truncated(train_eval['signal'], train_probs)
print ("AUC metric",AUC) 




"""
print("Make predictions on the test set")
test_probs = gd.predict_proba(test[features])[:,1] 
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("gd_xgboost_submission.csv", index=False)
"""
result = pd.DataFrame({'id': test.index})
test_probs = 0.2*ub.predict_proba(test[features])[:, 1]+0.8*gbm.predict(xgb.DMatrix(test[features]))

result=pd.DataFrame({"id": test["id"], "prediction": test_probs})
result.to_csv('UBoost_GB_XGB_ADA_classifier.csv', index=False, sep=',')


