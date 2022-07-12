import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
np.random.seed(10)



train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# clean and split data

# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# split data into train and test
test_id = test.ID
test = test.drop(["ID"],axis=1)

X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
print(X_train.shape, X_test.shape, test.shape)

## # Feature selection
clf = ExtraTreesClassifier(random_state=1729)
selector = clf.fit(X_train, y_train)
# clf.feature_importances_ 
fs = SelectFromModel(selector, prefit=True)

X_train = fs.transform(X_train)
X_test = fs.transform(X_test)
test = fs.transform(test)

print(X_train.shape, X_test.shape, test.shape)

#X_train=X_train.iloc[:,220:300].values
#X_test=X_test.iloc[:,220:300].values
#test=test.iloc[:,220:300].values

#X_train=X_train.values
#X_test=X_test.values
#test=test.values

## # Train Model
# classifier from xgboost

#from sklearn.linear_model import Ridge
#model = Ridge(alpha=0.1, random_state=8899) 

from sklearn.neighbors import KNeighborsRegressor 
np.random.seed(88)
model=KNeighborsRegressor(n_neighbors=15, weights='uniform', algorithm='auto', leaf_size=300, p=1, metric='minkowski', n_jobs=-1)


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train=mms.fit_transform(X_train)
X_test = mms.transform(X_test)
test = mms.transform(test)




print("Start modeling:")
model.fit(X_train, y_train)

# calculate the auc score
print("Roc AUC: ", roc_auc_score(y_test, model.predict(X_test),
              average='macro'))
              
## # Submission
probs = model.predict(test)

print("Save test prediction:")

submission = pd.DataFrame({"ID":test_id, "TARGET": probs})
submission.to_csv("z_knn_36Fea_v3_Apr23.csv", index=False)

print("Done")

