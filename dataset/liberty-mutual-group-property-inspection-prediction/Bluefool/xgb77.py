# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv",index_col=0)
test  = pd.read_csv("../input/test.csv",index_col=0)

labels = np.log(train.Hazard)
train.drop('Hazard', axis=1, inplace=True)

train_s = train
test_s = test

columns = train.columns
test_ind = test.index

train_s = np.array(train_s)
test_s = np.array(test_s)

# label encode the categorical variables
for i in range(train_s.shape[1]):
    if i in [3,4,5,6,7,8,10,11,14,15,16,19,21,27,28,29]:
        print(i)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
        train_s[:,i] = lbl.transform(train_s[:,i])
        test_s[:,i] = lbl.transform(test_s[:,i])

clf =KNeighborsClassifier(n_neighbors=15)
clf.fit(train_s,labels)

preds = np.exp(clf.predict_proba(test_s)[:,1])

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('knn_v1.csv')
  

