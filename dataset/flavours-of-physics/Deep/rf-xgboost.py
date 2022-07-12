import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from importlib.machinery import SourceFileLoader
folder = '../input/'
evaluation = SourceFileLoader("module.name", folder + "evaluation.py").load_module()
from types import MethodType

def predict_proba(name,clf,X):
    if name=='xgb':
        return clf.predict(xgb.DMatrix(X))
    else:
        return clf.predict_proba(X)[:,1]
    
xgb.Booster.predict_proba = predict_proba
    
def add_noise(array, level=0.40, random_seed=34):
    np.random.seed(random_seed)
    return level * np.random.random(size=array.size) + (1 - level) * array
    
def check_tests(clf,train,features,name):
    print('Checking tests for: ' + name + '...')
    # run the agreement test
    check_agreement = pd.read_csv(folder + 'check_agreement.csv', index_col='id')
    agreement_probs = predict_proba(name,clf,check_agreement[features])
    ks = evaluation.compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print('KS metric', ks, ks < 0.09)
    
    # perform evaluation on the training set itself
    train_eval = train[train['min_ANNmuon'] > 0.4]
    train_probs = predict_proba(name,clf,train_eval[features])
    AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
    print('AUC before noise: ', AUC)
    
    # test correlation with mass
    check_correlation = pd.read_csv(folder + 'check_correlation.csv', index_col='id')
    correlation_probs = predict_proba(name,clf,check_correlation[features])
    cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
    print('CvM metric', cvm, cvm < 0.002)
    
    # Add noise and repeat the test
    agreement_probs = add_noise(predict_proba(name,clf,check_agreement[features]))

    ks = evaluation.compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print('KS metric', ks, ks < 0.09)
    
    correlation_probs = add_noise(predict_proba(name,clf,check_correlation[features]))
    cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
    print('CvM metric', cvm, cvm < 0.002)
    
    train_eval = train[train['min_ANNmuon'] > 0.4]
    train_probs = add_noise(predict_proba(name,clf,train_eval[features]))
    AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
    print('AUC after noise: ', AUC)

print("Load the training/test data using pandas")
train = pd.read_csv(folder + "training.csv")
test  = pd.read_csv(folder + "test.csv")

print("Eliminate SPDhits, which makes the agreement check fail")
features = list(train.columns[1:-5])
num_estimators=500
print("Train a Random Forest model")
rf = RandomForestClassifier(verbose=0,n_estimators=num_estimators, n_jobs=-1, criterion="entropy", random_state=1)
rf.fit(train[features], train["signal"])

#check_tests(rf,train,features,'rf')
num_trees=1000
print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.1,
          "max_depth": 7,
          "min_child_weight": 10,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
         "seed": 1}

gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)
#check_tests(gbm,train,features,'xgb')

print("Make predictions on the test set")
test_probs = (0.6*rf.predict_proba(test[features])[:,1] +
              0.4*gbm.predict(xgb.DMatrix(test[features])))#/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission_750.csv", index=False)