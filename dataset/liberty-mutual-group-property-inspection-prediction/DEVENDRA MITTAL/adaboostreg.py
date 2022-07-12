# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier


FTRAIN = "../input/train.csv"
FTEST = "../input/test.csv"

preprocessing_columns = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']

def LoadData(test = False, suffle=False) :
    fName = FTEST if test else FTRAIN
    df = pd.read_csv(fName, header = 0)
    df = df.drop(['Id'], axis=1)
    le = preprocessing.LabelEncoder()
    for column in preprocessing_columns :
        df[column + '_new'] = le.fit_transform(df[column])
    df = df.drop(preprocessing_columns, axis=1)
    if not test :
        y = df['Hazard'].values
        X = df.drop(['Hazard'], axis = 1).values
        if suffle :
            X, y = shuffle(X, y, random_state=42)
    else :
        X = df.values
        y = None
    X = X.astype(np.float32)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    return X, y
    
def convertInt(x):
    try:
        return x.astype(int)
    except:
        return x

if __name__ == '__main__' :
    X, Y = LoadData() 
    estimator = 500
    clf = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = estimator, learning_rate = 0.1)
    clf.fit(X, Y)
    Xtest, y = LoadData(test = True)
    YPred = clf.predict(Xtest)
    Idtest = pd.read_csv(FTEST,header = 0)
    Idtest = Idtest['Id'].values
    output = pd.DataFrame(np.column_stack((Idtest, YPred)), columns = ['Id', 'Hazard'])
    output['Id'] = output['Id'].apply(convertInt)
    output.to_csv("AdaBoost" + str(estimator) + ".csv", index = False)
