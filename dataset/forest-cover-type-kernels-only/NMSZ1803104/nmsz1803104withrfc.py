import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#from sklearn.cross_validation import train_test_split

#from sklearn.svm import LinearSVC
train = pd.read_csv('../input/train.csv',  encoding='gbk')
test = pd.read_csv('../input/test.csv', encoding='gbk')
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
#train.drop(['Soil_Type15', "Soil_Type7"], inplace=True, axis=1)
#test.drop(['Soil_Type15', "Soil_Type7"], inplace=True, axis=1)
y = train["Cover_Type"]
train.drop(["Cover_Type"], inplace=True, axis=1)
x = train


#train_pre = train.columns[:10]
#train_scale = minmax_scale(train[train_pre].astype(float))
#train_post = train[train.columns[10:]]
#train_scale = pd.DataFrame(train_scale, columns=train_pre)
#train_new = pd.concat([train_scale, train_post], axis=1)

#test_pre = test.columns[:10]
#test_scale = minmax_scale(test[train_pre].astype(float))
#test_post = test[test.columns[10:]]
#test_scale = pd.DataFrame(test_scale, columns=train_pre)
#test_new = pd.concat([test_scale, test_post], axis=1)
#predictors = train_new.columns[:-1]

feed = 42
test_size = 0.25
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=feed)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
#knc=KNeighborsClassifier()
#knc.fit(X_train, y_train)
#dtc=DecisionTreeClassifier()
model=RandomForestClassifier()
#gbc=GradientBoostingClassifier()
#xgbc=XGBClassifier()
#model = XGBClassifier(learning_rate=0.3, n_estimators=350, max_depth=9, objective='multi:softmax', num_class=7)
#percentiles=range(1,100,2)
#results=[]
#for i in percentiles:
#	fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
#	X_train_fs=fs.fit(X_train,y_train)
#	scores=cross_val_score(dtc,X_train_fs,y_train,cv=5)
#	results=np.append(results,scores.mean())
#print(results)
#opt=np.where(results==results.max())[0]
#print('Optimal number of features %d' %percentiles[opt])

#xgbc.fit(X_train,y_train)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print ('Accuracy:',accuracy)
X_sub=test
X_sub=ss.transform(X_sub)
pred=model.predict(X_sub)
pred = pd.Series(pred, name='Cover_Type')
sub = pd.read_csv('../input/sample_submission.csv')
sub.drop('Cover_Type', axis=1, inplace=True)
sub = pd.concat([sub, pred], axis=1)
sub.to_csv('./NMSZ1803104.csv', index=False)
print('Saved.')