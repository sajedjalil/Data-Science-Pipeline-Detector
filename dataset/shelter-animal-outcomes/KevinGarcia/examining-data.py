import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/train.csv')

def encode_onehot(df, cols):
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def clean_data(df):
    df = df.drop(['AnimalID', 'Name', 'DateTime', 'OutcomeSubtype', 'AgeuponOutcome'], axis=1)
    df = encode_onehot(df, cols=['AnimalType', 'Breed', 'Color', 'SexuponOutcome'])
    return df

train_df = clean_data(train_df[:2000]).fillna(0)
test_df = clean_data(test_df[:2000]).fillna(0)

le = preprocessing.LabelEncoder()

X_train = train_df.drop(['OutcomeType'], axis=1).values
X_test = test_df.drop(['OutcomeType'], axis=1).values
y_train = le.fit_transform(train_df['OutcomeType'].values)
print(le.classes_)
y_test = le.fit_transform(test_df['OutcomeType'].values)
print(le.classes_)

clf = SVC(gamma=0.001)
clf.fit(X_train, y_train)

expected = y_test
predictions = clf.predict(X_test)
print(predictions)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predictions)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predictions))
