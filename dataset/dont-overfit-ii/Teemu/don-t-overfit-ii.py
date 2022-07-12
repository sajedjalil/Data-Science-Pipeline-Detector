#List of libraries and modules used (updated as we go ahead)
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Let's look first into the target variable: we alreayd its binary, but here we can see its mean is slightly skewed
train['target'].describe()
#sns.countplot(x='target',data=train)
#The variables itself are numeric - http://chris.friedline.net/2015-12-15-rutgers/lessons/python2/03-data-types-and-format.html
train.dtypes.value_counts()
#No NA values exist:
train.isnull().values.any()
# Let's see basic info
train.describe()
#Data is not skewed beside target variable. The reason may be small size of target-values in the Training set!
numeric_feats = train.dtypes[train.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.5]
skewed_feats 
############################################## LOGISTIC REGRESSION ##############################################
#Create variables and remove id & target columns:
# If doubts about axis 1 or 0 values - see link:
# https://cmdlinetips.com/2018/04/how-to-drop-one-or-more-columns-in-pandas-dataframe/
variables = train.drop(['id', 'target'], axis=1)
variables.shape
#Create target
target = train['target']
target.shape
#Let's split training set
X_train, X_test, y_train, y_test = train_test_split(variables, target, test_size=0.30,random_state=None)
# Let's fit the data to the basic-predictor:
basic =LogisticRegression(penalty='l1', C=0.12, solver='saga', warm_start=True)
basic.fit(X_train, y_train)
#Prediction using the basic-predictor
predictions = basic.predict(X_test)
#let's evaluate the "basic"-model
print(classification_report(y_test,predictions))
print("Accuracy:",accuracy_score(y_test, predictions))
#predict probabilities
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from matplotlib import pyplot
probs = basic.predict_proba(X_test)
probs = probs[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1 = f1_score(y_test, predictions)
auc = auc(recall, precision)
ap = average_precision_score(y_test, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
pyplot.plot(recall, precision, marker='.')
pyplot.show()

#Alternative ROC:
#auc = roc_auc_score(y_test, probs)
#print('AUC: %.3f' % auc)
#fpr, tpr, thresholds = roc_curve(y_test, probs)
#pyplot.plot([0, 1], [0, 1], linestyle='--')
#pyplot.plot(fpr, tpr, marker='.')
#pyplot.show()
# See more: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# Another very good one: https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

# Precision describes how good a model is at predicting the positive class
# Recall is the same as sensitivity
# F1-score: https://en.wikipedia.org/wiki/F1_score

# Let's re-fit with all the data on training set
basic =LogisticRegression(penalty='l1', C=0.1, solver='liblinear', warm_start=True)
basic.fit(variables, target)
#Based on the accuracy, let's add all the training set.
X_id = test['id']
X_pred = test.drop('id', axis=1)
y_pred = basic.predict(X_pred)
result = pd.DataFrame({'id': X_id, 'target': y_pred})
result.to_csv('result.csv', index = False)