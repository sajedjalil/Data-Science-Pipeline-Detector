"""
__author__ = Marjan Mazrouei Sebdani
"""


import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit
from sklearn import ensemble, svm, cross_validation
import matplotlib.pyplot as plt
from os import listdir


# load training and testing datasets
train = pd.read_csv(os.path.join('..', 'input', 'train_set.csv'), parse_dates=[2,])
test = pd.read_csv(os.path.join('..', 'input', 'test_set.csv'), parse_dates=[3,])
# attach testing data to training data
length_data = train.shape[0]
train['id'] = pd.Series(np.arange(-1,-length_data,-1))
test['cost']=0
train = train.append(test)

# Find the common variables in other ".csv" files 
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
# you need to add your path to directorty
path_to_dir = "../input/"
length_of_files = len(find_csv_filenames(path_to_dir))
#combine trianing file with  files based on common features
while(length_of_files>0):
	for files in find_csv_filenames(path_to_dir):
		length_of_files = length_of_files-1
		d= pd.read_csv(os.path.join('..', 'input', files))
		a = d.columns.values
		b = train.columns.values
		commonValues = list(set(a)& set(b))
		if (len(commonValues) >= int(1)):
			train = pd.DataFrame.merge(train, d, on = commonValues, how = 'left')
# divide quote_date to year, month, dayofyear, dayofweek and day
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['dayofyear'] = train.quote_date.dt.dayofyear
train['dayofweek'] = train.quote_date.dt.dayofweek
train['day'] = train.quote_date.dt.day

# drop quote_date and tube_assembly_id
train = train.drop(['quote_date', 'tube_assembly_id'], axis=1)
numcolumns = []
nonnumcolumns = []
# find numerical (numcolumns) and characteristic (nonumcolumns) columns
for i in range(len(train.columns)):
    if train.ix[:,i].dtypes != 'O':
        numcolumns.append(train.columns[i])
    else:
        nonnumcolumns.append(train.columns[i])

print("Numcolumns %s " % numcolumns)
print("Nonnumcolumns %s " % nonnumcolumns)
print("Nans before processing: \n {0}".format(train.isnull().sum()))
train[numcolumns] = train[numcolumns].fillna(-999999)
train[nonnumcolumns] = train[nonnumcolumns].fillna("NAvalue")
print("Nans after processing: \n {0}".format(train.isnull().sum()))
for col in nonnumcolumns:
    ser = train[col]
    counts = ser.value_counts().keys()
    threshold = 5
    if len(counts) > threshold:
        ser[~ser.isin(counts[:threshold])] = "rareValue"
    if len(counts) <= 1:
        print("Dropping Column %s with %d values" % (col, len(counts)))
        train = train.drop(col, axis=1)
    else:
        train[col] = ser.astype('category')

train = pd.get_dummies(train)
print("Size after dummies {0}".format(train.shape))

# seperate cost columns from training and testing data 
labels = train[train['id']<int(1)].cost.values
label_log = np.log1p(labels)
# drop cost columns from training and testing columns
Xtrain = train[train['id']< int(1)].drop(['cost'], axis=1)
Xtest = train[train['id']>= int(1)].drop(['cost'], axis=1)
allcolumns = Xtrain.columns.values

Xtrain1, label_log = shuffle(Xtrain, label_log, random_state=666)
clf = ensemble.RandomForestRegressor()
model = clf.fit(Xtrain1, label_log)

columns = []
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# find the 10 most important feature in our model
for f in range(len(importances)):
    print("%d. feature %d (%f), %s" % (f + 1, indices[f],importances[indices[f]], allcolumns[indices[f]]))
    columns.append(allcolumns[indices[f]])
    # Print only first 5 most important variables
    if len(columns) >=6:
        break

q = pd.qcut(labels, 5)
#fig = plt.figure()


# plot the most important features of the model

#f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1, figsize=(10, 8), sharex=True)
#pg = sns.barplot(x= q, y = Xtrain[columns[1]],data = Xtrain, palette="BuGn_d", ax=ax1)
#ax1.set_ylabel(train[columns[1]])
#pg = sns.barplot(x= q, y = Xtrain[columns[2]],data = Xtrain, palette="RdBu_r", ax=ax2)
#ax2.set_ylabel(train[columns[2]])
#sns.barplot(x= q, y = Xtrain[columns[3]], data = Xtrain , palette="BuGn_d", ax=ax3)
#ax3.set_ylabel(train[columns[3]])
#sns.barplot(x= q, y = Xtrain[columns[4]], data = Xtrain ,palette="RdBu_r", ax=ax4)
#ax4.set_ylabel(train[columns[4]])
#sns.barplot(x= q, y = Xtrain[columns[5]], data = Xtrain,palette="BuGn_d", ax=ax5)
#ax5.set_ylabel(train[columns[5]])


#sns.despine(bottom=True)
#plt.setp(f.axes, yticks=[])
#plt.savefig("marjan.png")	
model1 = clf.fit(Xtrain[columns], label_log)
preds = model1.predict(Xtest[columns])
model = clf.fit(Xtrain,label_log)
preds1 = model.predict(Xtest)
print("k-Fold RMSLE:")
cv_rmsle = cross_validation.cross_val_score(model1, Xtrain[columns], label_log, scoring='mean_squared_error')
cv_rmsle1 = cross_validation.cross_val_score(model, Xtrain, label_log, scoring='mean_squared_error')
print(cv_rmsle)
cv_rmsle = np.sqrt(np.abs(cv_rmsle))
cv_rmsle1 = np.sqrt(np.abs(cv_rmsle1))

print(cv_rmsle)
print("Mean in model with some features: " + str(cv_rmsle.mean()))
print("Mean in model with all features: " + str(cv_rmsle1.mean()))


preds = np.exp(preds)-1
print(preds)
preds1 = np.exp(preds1)-1
print(preds1)
preds = pd.DataFrame({"cost": preds})
preds.to_csv('marjan.csv', index=False)