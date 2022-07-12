### ~0.7 accuracy with full dataset, now testing subsets of columns.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def changedate(X):
    newdate = datetime.strptime(X, '%Y-%m-%d')
    utc = (newdate - datetime(1970,1,1)).total_seconds()
    return utc

df_train_raw = pd.read_csv("../input/train.csv")
df_test_raw = pd.read_csv("../input/test.csv")

#basecols = [col for col in df_train_raw.columns if col[-1] not in ['A', 'B']]
#base_a = basecols+[col for col in df_train_raw.columns if col[-1] == 'A']
#base_b = basecols+[col for col in df_train_raw.columns if col[-1] == 'B']
#%%

#convert dates to Unix-style timestamps
print ("Converting dates to Unix stamps")
df_train_raw['Quote_year'] = [int(dt.split('-')[0]) for dt in df_train_raw['Original_Quote_Date']]
df_train_raw['Quote_month'] = [int(dt.split('-')[1]) for dt in df_train_raw['Original_Quote_Date']]
df_train_raw['Quote_day'] = [int(dt.split('-')[2]) for dt in df_train_raw['Original_Quote_Date']]
df_train_raw['Original_Quote_Date']=[changedate(date) for date in df_train_raw['Original_Quote_Date']]
df_test_raw['Quote_year'] = [int(dt.split('-')[0]) for dt in df_test_raw['Original_Quote_Date']]
df_test_raw['Quote_month'] = [int(dt.split('-')[1]) for dt in df_test_raw['Original_Quote_Date']]
df_test_raw['Quote_day'] = [int(dt.split('-')[2]) for dt in df_test_raw['Original_Quote_Date']]
df_test_raw['Original_Quote_Date']=[changedate(date) for date in df_test_raw['Original_Quote_Date']]

#replace NaN entries in the frame
print ("Changing NaN values")
df_train_raw.fillna(-999, inplace = True)
df_test_raw.fillna(-999, inplace = True)

#convert categorical columns to numeric encodings
print ("Encoding categorical columns")
from sklearn import preprocessing

for f in df_train_raw.columns:
    if df_train_raw[f].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(list(df_train_raw[f].values) + list(df_test_raw[f].values))) # be sure to include BOTH frames' data....
        df_train_raw[f] = le.transform(list(df_train_raw[f].values))        # encode each frame separately
        df_test_raw[f] = le.transform(list(df_test_raw[f].values))          # naturally
#%%

df_train = df_train_raw
#df_train = df_train_raw[basecols]
#df_train = df_train_raw[base_a]
#df_train = df_train_raw[base_b]

df_test = df_test_raw
#df_test = df_test_raw[basecols]
#df_test = df_test_raw[base_a]
#df_test = df_test_raw[base_b]

#drop useless column
df_train = df_train.drop(['QuoteNumber'], axis=1)

#build test/train datasets
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, matthews_corrcoef, hamming_loss

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

clf1 = ExtraTreesClassifier()
clf2 = RandomForestClassifier()
clf3 = SGDClassifier()
clf4 = RidgeClassifier()
clf5 = LinearDiscriminantAnalysis()
clf6 = GradientBoostingClassifier(n_estimators=100)
clfZ = DecisionTreeClassifier()

#classifiers = [clf1, clf2, clf3, clf4, clf5]
classifiers = [clf6]

X_train = df_train.drop("QuoteConversion_Flag",axis=1).values
Y_train = df_train["QuoteConversion_Flag"].values
X_test  = df_test.drop("QuoteNumber",axis=1).copy().values

#print ("Rescaling data")
#MAscale = preprocessing.MaxAbsScaler()
#X_trainS = MAscale.fit_transform(X_train)
#X_testS  = MAscale.transform(X_test)

results=[]

#print("Filtering for features")
#sfm = SGDClassifier(loss='modified_huber', penalty='elasticnet').fit(X_trainS, Y_train)
#model = SelectFromModel(sfm, prefit=True)
#X_train_new = model.transform(X_train)


def score_clf(clf, X, Y):
    scores = cross_val_score(clf, X, Y, cv=KFold(5))
    clf.fit(X, Y)
    score = clf.score(X, Y)
    Y_pred = clf.predict(X)
    f1 = f1_score(Y, Y_pred)
    matt = matthews_corrcoef(Y,Y_pred)
    ham = hamming_loss(Y, Y_pred)
    results.append((clf, score, scores, (scores.mean(), 2*scores.std(), f1, matt, ham)))
    
for i in range(len(classifiers)):
    print(i)
    clf = classifiers[i]

#select from SGD model:
#    pipe = Pipeline([('sfm', SGDClassifier(loss='log', penalty='elasticnet')),
#                    ('clf', classifiers[i])])
    score_clf(clf, X_train, Y_train)

#    pred = clf.predict(X_testS)
#    df_test['QuoteConversion_Flag'] = pred
    
#    out = df_test[['QuoteNumber','QuoteConversion_Flag']]
#    out.to_csv('submission_4_'+str(i)+'.csv', index=False)

for r in results:
    print (r[3])
