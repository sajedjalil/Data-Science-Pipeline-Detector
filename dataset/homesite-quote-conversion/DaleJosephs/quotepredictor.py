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

basecols = [col for col in df_train_raw.columns if col[-1] not in ['A', 'B']]
base_a = basecols+[col for col in df_train_raw.columns if col[-1] == 'A']
base_b = basecols+[col for col in df_train_raw.columns if col[-1] == 'B']
#%%

#convert dates to Unix-style timestams
df_train_raw['Original_Quote_Date']=[changedate(date) for date in df_train_raw['Original_Quote_Date'].values.tolist()]
df_test_raw['Original_Quote_Date']=[changedate(date) for date in df_test_raw['Original_Quote_Date'].values.tolist()]

#replace NaN entries in the frame
df_train_raw.fillna(-1, inplace = True)
df_test_raw.fillna(-1, inplace = True)

#convert categorical columns to numeric encodings
from sklearn import preprocessing

for f in df_train_raw.columns:
    if df_train_raw[f].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(list(df_train_raw[f].values) + list(df_test_raw[f].values))) # be sure to include BOTH frames' data....
        df_train_raw[f] = le.transform(list(df_train_raw[f].values))        # encode each frame separately
        df_test_raw[f] = le.transform(list(df_test_raw[f].values))          # naturally

MAscale = preprocessing.MaxAbsScaler()
df_train_scaled = MAscale.fit_transform(df_train_raw)
df_test_scaled = MAscale.fit_transform(df_test_raw.drop(['QuoteNumber'], axis=1))
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
df_train_scaled = df_train_scaled.drop(['QuoteNumber'], axis=1)
print ('Foo!')

#build test/train datasets
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

clf1 = ExtraTreesClassifier()
clf2 = RandomForestClassifier()
clf3 = DecisionTreeClassifier()
clf4 = SGDClassifier()
clf5 = PassiveAggressiveClassifier()
clf6 = RidgeClassifier()
clf7 = LinearDiscriminantAnalysis()
clf8 = GaussianNB()


classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]

X_train1 = df_train.drop("QuoteConversion_Flag",axis=1).values
X_train2 = df_train_scaled.drop("QuoteConversion_Flag",axis=1).values
Y_train = df_train["QuoteConversion_Flag"].values
X_test  = df_test.drop("QuoteNumber",axis=1).copy().values

results=[]

def score_clf(clf, X1, X2, Y):
    scores1 = cross_val_score(clf, X1, Y, cv=KFold(Y_train.size, 5))
    scores2 = cross_val_score(clf, X2, Y, cv=KFold(Y_train.size, 5))
    clf.fit(X, Y)
    score1 = clf.score(X, Y)
    results.append((clf, score, scores1, scores2
                    (scores1.mean(), 2*scores1.std()),
                    (scores2.mean(), 2*scores2.std())))
    
for i in classifiers:
    print(i)
    #pipe = Pipeline([('pca',PCA(n_components=20)),
    #                 ('clf', i)])
    score_clf(i, X_train1, X_train2, Y_train)
    
for r in results:
    print ('{0}, {1} || {2}, {3}'.format(r[4][0],r[4][1],r[5][0],r[5][1]))

"""
    pred = clf.predict(X_test)
    df_test['QuoteConversion_Flag'] = pred
    
    out = df_test[['QuoteNumber','QuoteConversion_Flag']]
    out.to_csv('submission_2_'+str(i)+'.csv', index=False)
"""