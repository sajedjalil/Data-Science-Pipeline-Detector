# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# pandas
import pandas as pd
from pandas import Series,DataFrame
from pandas import get_dummies
# tensorflow
import tensorflow as tf

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

# machine learning
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# you could use Naive Bayes classification! just make sure it doesn't
# overfit...
# it's not reading the dataframe correctly.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#train_df.head()
#print('XXXXXXXXXXXXXXXSHADEXXXXXXXXXXX')
#train_df.info()
#print('XXXXXXXXXXXXXXXSHADEXXXXXXXXXXX')
#test_df.info()
#print('XXXXXXXXXXXXXXXSHADEXXXXXXXXXXX')
#print (train_df)
# now I'm on my way to making my first chart!!
# so the columns are: id, species, margin1, margin2, margin3
sns.factorplot('species','margin2', data = train_df,size = 4, aspect=3)
# well-conditioned data has zero mean! and equal variance!
data_norm = pd.DataFrame(train_df)
cols = train_df.columns

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
features = cols[0:194]
labels = data_norm["species"]
#for label in labels:
#    print(label)
#for label in labels:
#    print(label)
#Shuffle The data
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
X = data_norm.reindex(indices)[features]
y = labels
#y = data_norm.reindex(indices)[labels]
# One Hot Encode as a dataframe
#y = get_dummies(y)
# Generate Training and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)

# Convert to np arrays so that we can use with TensorFlow
X_train = np.array(X_train).astype(np.str)
X_test  = np.array(X_test).astype(np.str)
y_train = np.array(y_train).astype(np.str)
y_test  = np.array(y_test).astype(np.str)
#for feature in features:
#    print(train_df[feature])
#    train_df[feature] = (train_df[feature] - train_df[feature].mean())/train_df[feature].std()
#Check to make sure split still has 4 features and 3 labels
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#training_size = X_train.shape[1]
#test_size = X_test.shape[1]
num_features = 195
num_labels = 990


num_hidden = 10

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train_df, test_df)
train.head(1)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

clf = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)
name = clf.__class__.__name__
    
print("="*30)
print(name)
    
print('****Results****')
train_predictions = clf.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print("Accuracy: {:.4%}".format(acc))
train_predictions = clf.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print("Log Loss: {}".format(ll))
    
log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
log = log.append(log_entry)