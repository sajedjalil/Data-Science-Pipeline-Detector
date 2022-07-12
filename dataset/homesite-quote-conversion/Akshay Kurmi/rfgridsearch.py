import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import maxabs_scale
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def convert_features(train, test):
    groups = train.columns.to_series().groupby(train.dtypes).groups
    groups = {k.name : v for k,v in groups.items()}
    fields = groups.get('object')
    
    print("\nConverting", fields[0], flush=True)      
    def process_date(date, grpno):
        pattern = r"^(\d+)-(\d+)-(\d+)$"
        matchObj = re.match(pattern, date)
        result = matchObj.group(grpno)
        return int(result)
    train['Original_Quote_Year'] = [process_date(x, 1) for x in train['Original_Quote_Date']]
    train['Original_Quote_Month'] = [process_date(x, 2) for x in train['Original_Quote_Date']]
    train['Original_Quote_Day'] = [process_date(x, 3) for x in train['Original_Quote_Date']]    
    test['Original_Quote_Year'] = [process_date(x, 1) for x in test['Original_Quote_Date']]
    test['Original_Quote_Month'] = [process_date(x, 2) for x in test['Original_Quote_Date']]
    test['Original_Quote_Day'] = [process_date(x, 3) for x in test['Original_Quote_Date']]    
    
    for x in fields[1:]:        
        print("Converting", x)
        # For Y/N fields, replace the nulls and blanks with Q
        if len(train[x].unique()) <= 3:
            train[x] = train[x].where(pd.notnull(train[x]), 'Q')
            test[x] = test[x].where(pd.notnull(test[x]), 'Q')
            train[x] = train[x].where(train[x]!=' ', 'Q')
            test[x] = test[x].where(test[x]!=' ', 'Q')

        l = list(train[x].unique())
        l.extend(list(test[x].unique()))
        l = sorted(list(set(l)))
        
        le = LabelEncoder()
        le.fit(l)
        train[x] = le.transform(train[x])
        test[x] = le.transform(test[x])
        
    return train, test


def fill_missing(train, test):
    print("\nFilling Missing Data", flush=True)
    fields = []
    for column in train.columns.values:
        if train[column].isnull().values.any():
            fields.append(column)
    for column in test.columns.values:
        if test[column].isnull().values.any():
            if column not in fields:
                fields.append(column)
    
    # Attributes PersonalField84, PropertyField29 have missing values
    # PropertyField29 looks like yes/no and NA (0, 1, nan)
    train['PropertyField29'].fillna(0.5, inplace=True)
    test['PropertyField29'].fillna(0.5, inplace=True)
    # PersonalField84 has (nan, 1, 2, 3, 4, 5, 7, 8)
    train['PersonalField84'].fillna(0, inplace=True)
    test['PersonalField84'].fillna(0, inplace=True)
   
    return train, test
    
    
def drop_and_scale(train, test):
    print("Dropping Constant Fields", flush=True)
    const_cols = list()
    temp = train.loc[:, train.apply(pd.Series.nunique)!=1]
    for col in train.columns.values:
        if col not in temp.columns.values:
            const_cols.append(col)
    for col in const_cols:
        train.drop(col, inplace=True, axis=1)        
        test.drop(col, inplace=True, axis=1)
    
    print("Dropping Useless Fields", flush=True)
    QNos = test['QuoteNumber'].values
    train.drop('QuoteNumber', inplace=True, axis=1)
    test.drop('QuoteNumber', inplace=True, axis=1)
    Y_train = train['QuoteConversion_Flag'].values
    train.drop('QuoteConversion_Flag', inplace=True, axis=1)
    train.drop('Original_Quote_Date', inplace=True, axis=1)
    test.drop('Original_Quote_Date', inplace=True, axis=1)

    print("Scaling Features", flush=True)    
    X_train = maxabs_scale(train)
    X_test = maxabs_scale(test)
    
    return QNos, X_test, X_train, Y_train
    
    
def read_data():
    print("\nReading Data...", flush=True)
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    print("Done Reading Data...", flush=True)
    return train, test


def get_data():
    train, test = read_data()        
    train, test = convert_features(train.copy(deep=True), test.copy(deep=True))
    train, test = fill_missing(train.copy(deep=True), test.copy(deep=True))        
    QNos, X_test, X_train, Y_train = drop_and_scale(train.copy(deep=True), test.copy(deep=True))    
    return QNos, X_test, X_train, Y_train


def classify_RF(X_train, Y_train, X_test):
    global clf    
    print("\nRunning Random Forest", flush=True)    
    rfc = RandomForestClassifier(verbose=2)
    params = {'n_estimators':[50,100,200],
              'max_features':[3,6,9],
              'min_samples_leaf':[20,40,60],
              'n_jobs':[-1], 'class_weight':['balanced']}
    clf = GridSearchCV(rfc, params, scoring='roc_auc', cv=3, iid=False, verbose=2)
    clf.fit(X_train, Y_train)
    print("\nTraining Score :", clf.score(X_train, Y_train), flush=True)
    
    return clf.predict(X_test)
    

def main():
    QNos_test, X_test, X_train, Y_train = get_data()
    Y_test = classify_RF(X_train, Y_train, X_test)
    print("Generating Submission")
    submission = pd.read_csv("../input/sample_submission.csv")
    submission.QuoteConversion_Flag = Y_test
    submission.to_csv("RF_grid.csv", index=False)

if __name__ == "__main__":
    main()