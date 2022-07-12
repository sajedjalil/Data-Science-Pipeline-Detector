import pandas as pd
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from collections import Counter

def removeConstantDublicate(df):
    cols = df.columns.values.tolist()
    constantColumns = []
    unique_cols = []
    for index in range(0, df.shape[1]):
        #the number of unique elements in a column
        unique = len(pd.unique(df[cols[index]].ravel()))
        if unique <= 1:
            constantColumns.append(cols[index])
        else:
            unique_cols.append([unique, cols[index]])

    cols_set = possibleDublicates(unique_cols)
    duplicateColumns = findDublicates(df, cols_set)
    print ('Duplicated columns:', duplicateColumns)
    print ('Constant columns:', constantColumns)
    return (constantColumns + duplicateColumns)

def possibleDublicates(unique_cols):
    unique_cols.sort()
    poss_del = Counter([item[0] for item in unique_cols])
    poss_del_s = list(set(poss_del))
    cols_set = []
    for pos in poss_del_s:
        lst = []
        for item in unique_cols:
           if item[0] == pos:
               lst.append(item[1])
        cols_set.append(lst)
    return cols_set

##### Finding identical features
def findDublicates(df, cols_set):
    toRemove = []
    for cols in cols_set:
        features_pair = pairsFromSet(cols)
        for pair in features_pair:
            f1 = pair[0]
            f2 = pair[1]
            if (f1 not in toRemove) & (f2 not in toRemove):
                if (all(df[f1] == df[f2])):
                    #print (f1, "and", f2, "are equal.")
                    toRemove.append(f2)
    return toRemove

# creates all possible pair combination from a set
def pairsFromSet(source):
   result = []
   for p1 in range(len(source)):
      for p2 in range(p1 + 1, len(source)):
         result.append([source[p1], source[p2]])
   return result

def modelfit(alg, data, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    '''
    a variation of:
    http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    '''
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(data['x_train'][predictors], label=data['y_train'])
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(data['x_train'][predictors], data['y_train'], eval_metric='auc')
    #Predict training set:
    dtrain_predictions = alg.predict(data['x_train'][predictors])
    dtrain_predprob = alg.predict_proba(data['x_train'][predictors])[:,1]
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(data['y_train'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(data['y_train'], dtrain_predprob))
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp[0:20].plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    return alg

def prepareDataML(train_df, test_df):

    X = train_df.iloc[:,:-1]
    y = train_df.TARGET

    columnsDelete = removeConstantDublicate(X)

    X.drop(columnsDelete, axis=1, inplace=True)
    test_df.drop(columnsDelete, axis=1, inplace=True)

    ratio = y.value_counts() / float(y.size)
    print ('ratio of y: ', ratio)

    data = dict(
        x_train=X,
        x_test=test_df,
        y_train=y,
        y_test=[]
    )
    return data

def saveResults(dtrain_predprob, test_df, filename_output):
    df = pd.DataFrame(dtrain_predprob, columns=['TARGET'])
    print (df['TARGET'].mean())
    test_df = test_df.reset_index()
    df_res = pd.concat([test_df['ID'].astype(int), df], axis=1)
    df_res.to_csv(filename_output, index=False)

pd.set_option('display.precision', 5)

train_df = pd.read_csv("../input/train.csv", index_col=0)
test_df = pd.read_csv("../input/test.csv", index_col=0)

data = prepareDataML(train_df, test_df)
#TODO: find the parameters :)
xgbm = xgb.XGBClassifier(
   learning_rate=0.02,
   n_estimators=1500,
   max_depth=6,
   min_child_weight=1,
   gamma=0,
   subsample=0.9,
   colsample_bytree=0.85,
   objective= 'binary:logistic',
   nthread=4,
   scale_pos_weight=1,
   seed=27)

features = [x for x in data['x_train'].columns if x not in ['ID']]
alg = modelfit(xgbm, data, features)
dtrain_predprob = alg.predict_proba(data['x_test'][features])[:, 1]

saveResults(dtrain_predprob, test_df, 'submission.csv')