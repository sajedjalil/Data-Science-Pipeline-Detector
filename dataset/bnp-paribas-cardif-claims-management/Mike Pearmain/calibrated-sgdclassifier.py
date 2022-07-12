import pandas as pd
import numpy as np
import csv
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

# A fork of 'simple SGDClassifier' to add calibration example and change
# loss and prediction.

def clean_data(train):
    for row in train:
        if train[row].dtype == 'object':
                tmp_values= pd.factorize(train[row])
                train[row]=tmp_values[0]
        else:
                tmp_len = len(train[row].isnull())
                if tmp_len>0:
                    train.loc[train[row].isnull()] = train[row].mean()


if __name__ == "__main__":

    print ("Read Data")

    train=pd.read_csv("../input/train.csv")
    train_labels=train['target']
    train = train.drop(['ID','target'],axis=1)

    test=pd.read_csv("../input/test.csv")
    test_id=test['ID']
    test = test.drop(['ID'],axis=1)

    print ("Clean Data")
    clean_data(train)
    clean_data(test)
    print ("Train")
    sgd = SGDClassifier(loss="log",
                        penalty="elasticnet", 
                        n_jobs = -1,
                        n_iter = 100,
                        random_state = 123)
    model_calib = CalibratedClassifierCV(base_estimator=sgd, cv=5, method='isotonic')
    model_calib.fit(np.array(train.values),np.array(train_labels.values))
    print ("Predict")
    test_labels=model_calib.predict_proba(np.array(test.values))[:,1]

    predictions_file = open("simple_sgd_result.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "PredictedProb"])
    open_file_object.writerows(zip(test_id,test_labels))
    predictions_file.close()

    print ("Done"),
