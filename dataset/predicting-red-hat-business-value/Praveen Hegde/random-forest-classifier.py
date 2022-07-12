import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn import cross_validation

train = pd.read_csv('../input/act_train.csv',dtype={'people_id': np.str,
          						   'activity_id': np.str,
				                           'outcome': np.int8},
						    parse_dates=['date'])

Y = train['outcome']

train.drop('activity_id',axis=1,inplace=True)
train.drop('date',axis=1,inplace=True)
train.drop('outcome',axis=1,inplace=True)


test = pd.read_csv('../input/act_test.csv',dtype={'people_id':np.str,
							 'activity_id':np.str},
						  parse_dates=['date'])
act_id = test['activity_id']

test.drop('activity_id',axis=1,inplace=True)
test.drop('date',axis=1,inplace=True)

people = pd.read_csv('../input/people.csv',dtype={'people_id':np.str,
						  'activity_id':np.str,
						  'char_38': np.int32},								                             
					   parse_dates=['date'])

train['activity_category'] = train['activity_category'].str.lstrip('type ').astype(np.int32)
test['activity_category'] = test['activity_category'].str.lstrip('type ').astype(np.int32)

for i in range(1,11):
    charMax = train['char_'+str(i)].value_counts().idxmax()
    train['char_'+str(i)].fillna(charMax, inplace=True)
    train['char_'+str(i)] = train['char_'+str(i)].str.lstrip('type ').astype(np.int32)
    test['char_'+str(i)].fillna(charMax, inplace=True)
    test['char_'+str(i)] = test['char_'+str(i)].str.lstrip('type ').astype(np.int32)

people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
people.drop('date',axis=1,inplace=True)

for i in range(1, 10):
	people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
for i in range(10, 38):
	people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)


train = pd.merge(train,people,how='left',on='people_id',left_index=True)
test = pd.merge(test,people,how='left',on='people_id',left_index=True)

del people

train.drop('people_id',axis=1,inplace=True)
test.drop('people_id',axis=1,inplace=True)

X = train.as_matrix()
testX = test.as_matrix()

dt_regressor = DecisionTreeRegressor()
rf_regressor200 = RandomForestClassifier(n_estimators = 200)
log_regressor = linear_model.LogisticRegression()
mlp_regressor = MLPClassifier(activation='logistic')

#score0 = cross_validation.cross_val_score(rf_regressor1,X,Y,cv=5)
#score1 = cross_validation.cross_val_score(rf_regressor200,X,Y,cv=5)
#score2 = cross_validation.cross_val_score(log_regressor,X,Y,cv=5)
#score3 = cross_validation.cross_val_score(dt_regressor,X,Y,cv=5)
#score4 = cross_validation.cross_val_score(mlp_regressor,X,Y,cv=5)

#print "rf1: mean="+str(np.mean(score0))+" std="+str(np.std(score0))
#print "rf200: mean="+str(np.mean(score1))+" std="+str(np.std(score1))
#print "log_reg: mean="+str(np.mean(score2))+" std="+str(np.std(score2))
#print "dt_reg: mean="+str(np.mean(score3))+" std="+str(np.std(score3))
#print "mlp_reg: mean="+str(np.mean(score4))+" std="+str(np.std(score4))

rf_regressor200.fit(X,Y)
predict = rf_regressor200.predict(testX)

wrt = True
if wrt:
	submit = open('submit_rfc.csv','w')
	submit.write('activity_id,outcome\n')
	for i in range(0,len(predict)):
		submit.write(str(act_id[i])+","+str(predict[i])+"\n")
	submit.flush()



