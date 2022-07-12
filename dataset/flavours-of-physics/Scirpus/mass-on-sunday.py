import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_training_data(nonFeatures):
    mms = MinMaxScaler()
    logfeatures = []
    train = pd.read_csv('../input/training.csv')
    train1 = train
    train2 = train.copy()
    train1['FlightDistanceE'] = train1['FlightDistance']/train1['FlightDistanceError']
    y1 = mms.fit_transform((1./(50*math.sqrt(2*math.pi)))*np.exp(-((1776.82-train1.mass)**2)/(2*(50**2))))
    train1.drop(['id',
                'production',
                'signal',
                'mass',
                'min_ANNmuon'], inplace=True, axis=1)

    train1.drop(nonFeatures, inplace=True, axis=1)

    features = list(train1.columns)
    train2['FlightDistanceE'] = train2['FlightDistance']/train2['FlightDistanceError']
    y2 = mms.fit_transform((1./(50*math.sqrt(2*math.pi)))*np.exp(-((1776.82-train2.mass)**2)/(2*(50**2))))
    train2 = train2[features]

    for col in train1.columns:
        if(((train1[col].max() - train1[col].min()) > 10)):
            train1[col] = train1[col]-train1[col].min()
            logfeatures.append(col)
            train1[col] = np.log1p(train1[col])

    for col in logfeatures:
        train2[col] = train2[col]-train2[col].min()
        train2[col] = np.log1p(train2[col])

    print(logfeatures)
    print(features)

    return (logfeatures,
            features,
            np.array(train1[features]),
            np.array(y1),
            np.array(train2[features]),
            np.array(y2))


def get_test_data(logfeatures, features):
    test = pd.read_csv('../input/test.csv')
    test['FlightDistanceE'] = (test['FlightDistance']/ test['FlightDistanceError'])
    ids = test['id']
    test = test[features]
    for col in logfeatures:
        test[col] = test[col]-test[col].min()
        test[col] = np.log1p(test[col])
    return ids, np.array(test)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def grab_data(nonFeatures):
    (logfeatures,
     features,
     train1,
     trainlabel1,
     train2,
     trainlabel2) = get_training_data(nonFeatures)
    print('Train1 Data shape:', train1.shape)
    print('Train2 Data shape:', train2.shape)
    testids, test = get_test_data(logfeatures, features)
    print('Test Data shape:', test.shape)

    train1, scaler = preprocess_data(train1)
    train2, scaler = preprocess_data(train2, scaler)
    test, scaler = preprocess_data(test, scaler)

    return (logfeatures,
            features,
            scaler,
            train1,
            train2,
            trainlabel1,
            trainlabel2,
            testids,
            test)


def main():

    nonFeatures = ['FlightDistanceError',
                   'IPSig',
                   'isolationc',
                   'iso',
                   'SPDhits'
                   ]

    params = {}
    params["objective"] = "reg:logistic"
    params["eta"] = 0.01
    params["subsample"] = 0.6
    params["colsample_bytree"] = 0.6
    params["silent"] = 1
    params["eval_metric"] = 'auc'
    num_rounds = 500

    (logfeatures,
     features,
     scaler,
     train1,
     train2,
     trainlabel1,
     trainlabel2,
     testids,
     test) = grab_data(nonFeatures)

    xgtrain = xgb.DMatrix(train1, label=trainlabel1)
    xgval = xgb.DMatrix(train2, label=trainlabel2)
    xgtest = xgb.DMatrix(test)
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    params["min_child_weight"] = 100
    params["max_depth"] = 7
    params["scale_pos_weight"] = 1
    model = xgb.train(params, xgtrain, num_rounds, watchlist)
    predictions = model.predict(xgtest)
    submission = pd.DataFrame({"id": testids, "prediction": predictions})
    submission.to_csv("signalisforthebirds.csv", index=False)


if __name__ == "__main__":
    main()
