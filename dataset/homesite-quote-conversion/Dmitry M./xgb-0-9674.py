# feature tuning by https://www.kaggle.com/yangnanhai/homesite-quote-conversion/keras-around-0-9633
# and XGBoost by https://www.kaggle.com/sushize/homesite-quote-conversion/xgb-stop/run/104479

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# XGBoost params:
def get_params():
    #
    params = {}
    params["objective"] = "binary:logistic"
    params["booster"] = "gbtree"
    params["eval_metric"] = "auc"
    params["eta"] = 0.01 # 0.06, #0.01,
    #params["min_child_weight"] = 240
    params["subsample"] = 0.75
    params["colsample_bytree"] = 0.68
    params["max_depth"] = 7
    plst = list(params.items())
    #
    return plst


golden_feature=[("CoverageField1B","PropertyField21B"),
                ("GeographicField6A","GeographicField8A"),
                ("GeographicField6A","GeographicField13A"),
                ("GeographicField8A","GeographicField13A"),
                ("GeographicField11A","GeographicField13A"),
                ("GeographicField8A","GeographicField11A")]



def load_data(need_normalize = True):
    train=pd.read_csv("../input/train.csv")
    test=pd.read_csv("../input/test.csv")


    train = train.drop(['QuoteNumber','PropertyField6', 'GeographicField10A'], axis=1)

    submission=pd.DataFrame()
    submission["QuoteNumber"]= test["QuoteNumber"]

    test = test.drop(['QuoteNumber','PropertyField6', 'GeographicField10A'],axis=1)
    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    train = train.drop('Original_Quote_Date', axis=1)
    train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
    train['weekday'] = [train['Date'][i].dayofweek for i in range(len(train['Date']))]

    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
    test = test.drop('Original_Quote_Date', axis=1)
    test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
    test['weekday'] = [test['Date'][i].dayofweek for i in range(len(test['Date']))]

    train = train.drop('Date', axis=1)
    test = test.drop('Date', axis=1)

    #fill na
    train = train.fillna(-1)
    test = test.fillna(-1)

    for f in test.columns:# train has QuoteConversion_Flag
        if train[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f])+list(test[f]))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))

    train_y=train['QuoteConversion_Flag'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y).astype(np.int32)
    train=train.drop('QuoteConversion_Flag',axis=1)

    #add golden feature:
    for featureA,featureB in golden_feature:
        train["_".join([featureA,featureB,"diff"])]=train[featureA]-train[featureB]
        test["_".join([featureA,featureB,"diff"])]=test[featureA]-test[featureB]

    print ("processsing finished")

    train = np.array(train)
    train = train.astype(np.float32)
    test=np.array(test)
    test=test.astype(np.float32)
    if need_normalize:
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

    return [(train,train_y),(test,submission)]

print('Loading data...')


datasets=load_data()

X_train, y_train = datasets[0]
X_test, submission = datasets[1]

#Now we have fully tuned dataset

# convert data to xgb data structure
xgtrain = xgb.DMatrix(X_train, y_train)
xgtest = xgb.DMatrix(X_test)


boost_round = 5 #1800 CHANGE THIS BEFORE START
clf = xgb.train(get_params(),xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)

#Make predict
test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
# Save results
#
predictions_file = open("xgb_res.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["QuoteNumber", "QuoteConversion_Flag"])
open_file_object.writerows(zip(submission["QuoteNumber"].values, test_preds))
predictions_file.close()
#
print('Done.')