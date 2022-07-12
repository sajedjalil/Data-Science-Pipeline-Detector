# 1. based on original script by: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#    http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html
# 2. based on original script by: https://www.kaggle.com/jiweiliu, FTRL starter code
# 3. based on original script by: Lars Buitinck <L.J.Buitinck@uva.nl>
#    http://scikit-learn.org/stable/auto_examples/text/hashing_vs_dict_vectorizer.html?wb48617274=67251A88#example-text-hashing-vs-dict-vectorizer-py
# 4. based on original script by Harsh, Random Forest
#    https://www.kaggle.com/harshsinha/springleaf-marketing-response/rnadom-forest/run/51121

import pandas as pd
import numpy as np

#########################################################################################################
# feature selection: black list 
mixed_types = (8,9,10,11,12,43,196,214,225,228,229,231,235,238, 73, 40, 41, 42, 43, 39, 38, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 21, 20, 19, 18, 17, 8, 9, 10, 11, 12, 75, 44, 404, 305, 283, 222, 202, 204, 216, 217, 466, 467, 493)
black_list = (83, 84, 79, 80, 77, 78, 76, 69, 70, 71, 72, 68, 67, 66, 63, 62, 61, 60, 59, 58, 114, 75, 44, 404, 305, 283, 222, 202, 204, 216, 217, 466, 467, 493, 188, 189, 190, 454, 428, 427, 244, 266, 191, 217, 216, 204, 202, 75, 44, 73, 157, 158, 176, 156, 159, 166, 167, 168, 169, 177, 178, 179, 207, 213, 214, 840, 8, 9, 10, 11, 12, 43, 196, 229, 239)
black_list += mixed_types

columns = []
black_list_columns = []
for i in range(1,1935):
    if i not in black_list:
        columns.append(i)
    else:
        black_list_columns.append(i)

columns.sort()
black_list_columns.sort()
columns = [str(n).zfill(4) for n in columns]
columns = ['VAR_' + n for n in columns] 
columns.remove('VAR_0240')
columns.remove('VAR_0218')
columns.append('target')
columns.insert(0,'ID')

black_list_columns = [str(n).zfill(4) for n in black_list_columns]
black_list_columns = ['VAR_' + n for n in black_list_columns] 

#########################################################################################################
# get train data
n = 50000 # read from train
train1 = pd.read_csv("../input/train.csv", nrows = n - 1)# , usecols=columns
train2 = pd.read_csv("../input/train.csv", skiprows = range(1, 2*n), nrows= n)

train1.drop(black_list_columns, axis=1, inplace=True)
train2.drop(black_list_columns, axis=1, inplace=True)
columns.remove('target')
print("train1: %s" % (str(train1.shape)))
print("train2: %s" % (str(train2.shape)))
#submission_test = pd.read_csv("../input/test.csv", nrows=t)#, usecols=columns,
#submission_test.drop(black_list_columns, axis=1, inplace=True)
submission = 'submission.csv'
#print("test: %s" % (str(submission_test.shape)))

# fill NaN
#train = train.fillna(0)
#submission_test = submission_test.fillna(0)

# feature selection: use only numeric features
#numeric_include = ['int16', 'int32', 'float16', 'int64']
#numeric_exclude = ['float32', 'float64']
#train.select_dtypes(exclude=numeric_exclude)
#submission_test.select_dtypes(exclude=numeric_exclude)
#train = train.select_dtypes(include=numeric_include)
#submission_test = submission_test.select_dtypes(include=numeric_include)

# save IDs for submission
#ids = submission_test.values[:,0]

# remove IDs from train and test
train1.drop('ID', axis=1, inplace=True)
train2.drop('ID', axis=1, inplace=True)
#submission_test.drop('ID', axis=1, inplace=True)
print("train1: %s" % (str(train1.shape)))
print("train2: %s" % (str(train2.shape)))

#########################################################################################################
'''
# test hasher
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import re

hasher = FeatureHasher(n_features=10, non_negative=True, input_type='string')
train = hasher.fit_transform(train)
print(train)
'''

#LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#le.fit(X_train)
#X_train = le.transform(X_train)
#X_test = le.transform(X_test)
binarizer = preprocessing.Binarizer()

#########################################################################################################
# pre-processing: convert categorial to numeric
def to_categorial(df):
    """ Convert DataFrame non numeric columns to categorial indexes """
    for col in df:
        nans = df[col].isnull().sum()
        
        if not np.isreal(df[col][0]):
            #print(df[col].describe())
            #df[col] = pd.Categorical.from_array(df[col]).codes  

            if nans > 0:
                df[col] = df[col].fillna('Void')
                #df[col] = df[col].convert_objects(convert_numeric=False)
                #print(df[col].describe())
                #print("Str NaN: " + str(nans) + ", " + str(col))
            
            df[col] = df[col].astype(str)    
            le.fit(df[col])
            df[col] = le.transform(df[col])
        else:
            if nans > 0:
                df[col] = df[col].fillna(0)
                #df[col] = df[col].convert_objects(convert_numeric=False)
                #print(df[col].describe())
                #print("Numeric NaN: " + str(nans) + ", " + str(col))
    
    return df    
    
train1 = to_categorial(train1)
train2 = to_categorial(train2)

#split
from sklearn.cross_validation import train_test_split
train1, test1 = train_test_split(train1, test_size=0.1)
train2, test2 = train_test_split(train2, test_size=0.1)
train1 = train1.values
train2 = train2.values
test1 = test1.values
test2 = test2.values

#slice
X_train1 = train1[:,:-1]
X_train2 = train2[:,:-1]
X_test1 = test1[:,:-1]
X_test2 = test2[:,:-1]
y_train1 = train1[:,-1].astype(int)
y_train2 = train2[:,-1].astype(int)
y_test1 = test1[:,-1].astype(int)
y_test2 = test2[:,-1].astype(int)

#########################################################################################################
#binarization
binarizer.fit(X_train1)
X_train1 = binarizer.transform(X_train1)
X_train2 = binarizer.transform(X_train2)
X_test1 = binarizer.transform(X_test1)
X_test2 = binarizer.transform(X_test2)

# feature selection: Variance 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit(X_train1, y_train1)
X_train1 = sel.transform(X_train1)
X_train2 = sel.transform(X_train2)
X_test1 = sel.transform(X_test1)
X_test2 = sel.transform(X_test2)

print("\nfeature selection: Variance")
print("X_train1: %s" % (str(X_train1.shape)))
print("X_train2: %s" % (str(X_train2.shape)))
print("X_test1: %s" % (str(X_test1.shape)))
print("X_test2: %s" % (str(X_test2.shape)))
#print("submission_test: %s" % (str(submission_test.shape)))

# feature selection: Tree-based 
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(X_train1, y_train1)
X_train1 = etc.transform(X_train1)
X_train2 = etc.transform(X_train2)
X_test1 = etc.transform(X_test1)
X_test2 = etc.transform(X_test2)
#submission_test = etc.transform(submission_test)
print("\nfeature selection: Tree-based")
print("X_train1: %s" % (str(X_train1.shape)))
print("X_train2: %s" % (str(X_train2.shape)))
print("X_test1: %s" % (str(X_test1.shape)))
print("X_test2: %s" % (str(X_test2.shape)))
#print("submission_test: %s" % (str(submission_test.shape)))

# feature selection: Univariate
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_train1, y_train1)
X_train1 = min_max_scaler.transform(X_train1)
X_train2 = min_max_scaler.transform(X_train2)
X_test1 = min_max_scaler.transform(X_test1)
X_test2 = min_max_scaler.transform(X_test2)
#submission_test = min_max_scaler.transform(submission_test)

select = SelectKBest(chi2, k=200)
select.fit(X_train1, y_train1)
X_train1 = select.transform(X_train1)
X_train2 = select.transform(X_train2)
X_test1 = select.transform(X_test1)
X_test2 = select.transform(X_test2)
#submission_test = select.transform(submission_test)

print("\nfeature selection: Univariate")
print("X_train1: %s" % (str(X_train1.shape)))
print("X_train2: %s" % (str(X_train2.shape)))
print("X_test1: %s" % (str(X_test1.shape)))
print("X_test2: %s" % (str(X_test2.shape)))
#print("submission_test: %s" % (str(submission_test.shape)))

#########################################################################################################
# Create classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier

lr = LogisticRegression()
gnb = GaussianNB()
#svc = LinearSVC(C=1.0) #'LinearSVC' object has no attribute 'predict_proba'
#rfc = RandomForestClassifier(n_estimators=100)
#bag = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
#knn = KNeighborsClassifier(1)
#mnb = MultinomialNB() #positive only
#etc = ExtraTreesClassifier()
xgb = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05)
gbc = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)

#########################################################################################################
# evaluate    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
clf1 = lr
clf2 = gbc
clf1.fit(X_train1, y_train1)
clf2.fit(X_train2, y_train2)
predictions1 = clf1.predict(X_test1)
predictions2 = clf2.predict(X_test2)
print ('\nAUC 1', roc_auc_score(y_test1, predictions1))
print ('RMSE 1', mean_squared_error(y_test1, predictions1))
print ('\nAUC 2', roc_auc_score(y_test2, predictions2))
print ('RMSE 2', mean_squared_error(y_test2, predictions2))
#print('labels order',clf.classes_)

#########################################################################################################
# submission
print("\nWrite to file")
import random
from datetime import datetime

with open(submission, 'w') as outfile:
    outfile.write('ID,target\n')
    #for index, row in submission_test.iterrows():
    t = 145232
    chunk = 10000
    to_read = chunk
    skip = 0
    
    while t > 0:
        t -= chunk
        if t < 0:
            to_read += chunk
            t = 0
        
        #print("t:", str(t), "to_read:", str(to_read), "skip:", skip)   
        if skip > 0 and skip <= 145232:
            submission_test = pd.read_csv("../input/test.csv", skiprows=range(1,skip), nrows=to_read)
        elif skip == 0:
            submission_test = pd.read_csv("../input/test.csv", nrows=to_read - 1)
        
        skip += chunk
        
        submission_test.drop(black_list_columns, axis=1, inplace=True)
        #print(submission_test.describe())
        #print(submission_test.shape)
        
        #submission_test = submission_test.fillna(0)
        ids = submission_test.values[:,0]
        submission_test.drop('ID', axis=1, inplace=True)        
        submission_test = to_categorial(submission_test)
        submission_test = binarizer.transform(submission_test)
        submission_test = sel.transform(submission_test)
        submission_test = etc.transform(submission_test)
        submission_test = min_max_scaler.transform(submission_test)
        submission_test = select.transform(submission_test)

        count = 0
        for row in submission_test:
            p1 = clf1.predict_proba(row)
            p2 = clf2.predict_proba(row)
            if p1[0][1] > p2[0][1]:
                p = p1
            else:
                p = p2

            #outfile.write('%s,%s\n' % (ids[index], str(p[0][1])))
            outfile.write('%s,%s\n' % (ids[count], str(p[0][1])))
            #print('%s,%s\n' % (ids[count], str(p[0][1])))
            #if count % 10000 == 0:
                #print('%s\tlap: %d' % (datetime.now(), count))
            count += 1
        
    print('%s\tfinish: %d' % (datetime.now(), count))
    outfile.close()

#########################################################################################################   
#Warehouse 
'''
#submission backup
# submission
print("\nWrite to file")
import random
from datetime import datetime
count=0

with open(submission, 'w') as outfile:
    outfile.write('ID,target\n')
    #for index, row in submission_test.iterrows():
    for row in submission_test:
        p = clf.predict_proba(row)
        #outfile.write('%s,%s\n' % (ids[index], str(p[0][1])))
        outfile.write('%s,%s\n' % (ids[count], str(p[0][1])))
        if count % 10000 == 0:
            print('%s\tlap: %d' % (datetime.now(), count))
        count += 1
    print('%s\tfinish: %d' % (datetime.now(), count))
    outfile.close()

'''

'''
# hash trick
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=10, non_negative=True, input_type='string')
train = hasher.transform(train)
print(train)
#test = hasher.transform(test)
#X_train = hasher.transform(X_train)
#y_train = hasher.transform(y_train)
#submission_test = hasher.transform(submission_test)
'''

'''
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import re

def token_freqs(doc):
    """Extract a dict mapping tokens from doc to their frequencies."""
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq

def tokens(doc):
    """Extract tokens from doc.

    This uses a simple regex to break strings into tokens. For a more
    principled approach, see CountVectorizer or TfidfVectorizer.
    """
    return (tok.lower() for tok in re.findall(r"\w+", doc))    

'''

'''
tmp = train['VAR_0001']
#print(pd.get_dummies(tmp))
tmp = pd.Categorical.from_array(tmp).codes
#print(tmp.astype('category'))
print(tmp)
#print(train.applymap(np.isreal))
#print(tmp.T.to_dict().values())
#tmp = tmp.convert_objects(convert_dates=True, convert_numeric=False, convert_timedeltas=True, copy=True)
#train = train.replace('%','',regex=True).astype('float')/100
#from sklearn.feature_extraction import DictVectorizer
#vectorizer = DictVectorizer(sparse = False)
#vectorizer.transform(tmp)

#print(tmp.T.to_dict())

vectorizer = DictVectorizer(sparse=False)
#vectorizer.fit(token_freqs(d) for d in tmp)
vectorizer.fit_transform(tmp)
#tmp = vectorizer.transform(tmp)

#hasher = FeatureHasher(n_features=10, non_negative=True, input_type='string')
#tmp = hasher.fit_transform(tmp)
print(tmp)
'''    

'''
# test several classifiers
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print (name, ' roc_auc_score', roc_auc_score(y_test, predictions))
    print (name, 'RMSE', mean_squared_error(y_test, predictions))
'''  
