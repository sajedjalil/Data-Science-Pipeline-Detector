# feature tuning by https://www.kaggle.com/yangnanhai/homesite-quote-conversion/keras-around-0-9633
# and XGBoost by https://www.kaggle.com/sushize/homesite-quote-conversion/xgb-stop/run/104479

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



golden_feature=[("CoverageField1B","PropertyField21B"),
                ("GeographicField6A","GeographicField8A"),
                ("GeographicField6A","GeographicField13A"),
                ("GeographicField8A","GeographicField13A"),
                ("GeographicField11A","GeographicField13A"),
                ("GeographicField8A","GeographicField11A")]



def load_data(need_normalize = True):
    train=pd.read_csv("../input/train.csv")
    test=pd.read_csv("../input/test.csv")

    print ("processsing started")

    for f in test.columns:# train has QuoteConversion_Flag
        if train[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train[f])+list(test[f]))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))


    #add golden feature:
    for featureA,featureB in golden_feature:
        train["_".join([featureA,featureB,"diff"])]=train[featureA]-train[featureB]
        test["_".join([featureA,featureB,"diff"])]=test[featureA]-test[featureB]

    print ("processsing finished")

    train.to_csv('train.csv')
    test.to_csv('test.csv')
    

print('Loading data...')


datasets=load_data()

