import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gc
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel

#Read Input Data
#train = pd.read_csv('input/train1.csv')
#test = pd.read_csv('input/test1.csv')

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)


#Extrate Date Details
train['Original_Quote_Date_Typed'] = pd.to_datetime(train.Original_Quote_Date)
train['month'] = train.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%m'))
train['day_of_week'] = train.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%w'))

test['Original_Quote_Date_Typed'] = pd.to_datetime(test.Original_Quote_Date)
test['month'] = test.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%m'))
test['day_of_week'] = test.Original_Quote_Date_Typed.apply(lambda x:x.strftime('%w'))

#Drop Quote Date
train.drop(['Original_Quote_Date'],axis=1,inplace=True)
test.drop(['Original_Quote_Date'],axis=1,inplace=True)

#Fill Nan Values
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

#Label Nominal Values (Non Numeric - which is dtype=Object)
for f in train.columns:
    if train[f].dtype=='object':     
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train[f].values) + list(test[f].values)))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

#Prepare Complete Training & Test Data for Classifier         
Y_train = train["QuoteConversion_Flag"]
X_train = train.drop(["QuoteConversion_Flag","QuoteNumber","Original_Quote_Date_Typed"],axis=1)
X_test  = test.drop(["QuoteNumber","Original_Quote_Date_Typed"],axis=1).copy()

# Select the importance Features in the tree classifier 

#clf = ExtraTreesClassifier()
clf=RandomForestClassifier()
clf = clf.fit(X_train, Y_train)
clf.feature_importances_  

# Select the importance features.  All the features greater than the mean of feature weightage

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

mean = importances.mean()

indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")
'''
special_indices=[]
for f in range(X_train.shape[1]):    
    if importances[indices[f]] > mean:
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
'''
special_features=[]
for f in range(X_train.shape[1]):
    if importances[f] > mean:
#        print X_train.ix[:,f].name
        special_features.append(X_train.ix[:,f].name)

X_special_train = X_train[special_features]
X_special_test  = X_test[special_features]

print (X_special_train.shape)
print (X_special_test.shape)

# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X_train.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X_train.shape[1]), indices)
#plt.xlim([-1, X_train.shape[1]])
#plt.show()

#Run RandomForrestClassifier in the selected Special Features
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_special_train, Y_train)
#rf.fit(X.values, Y.values)

#Predict the Output
Y_test=rf.predict(X_special_test)

#Create Submission
submission = pd.DataFrame()
submission["QuoteNumber"]          = test["QuoteNumber"]
submission["QuoteConversion_Flag"] = Y_test
submission.to_csv('homesite_special.csv', index=False)

