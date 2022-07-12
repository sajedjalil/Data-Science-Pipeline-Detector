# import tools
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np


# import data
trainDataX = pd.read_csv('../input/train.csv', nrows = 27000000)
testDataX = pd.read_csv('../input/test.csv')

# select data we want
trainDataY = trainDataX.Demanda_uni_equil
trainDataX = trainDataX.iloc[:,0:6]
testDataX = testDataX.iloc[:,1:7]

###############################################################
### these line is used for cross validation testing         
###############################################################
#X_train, X_test, y_train,y_test = cross_validation.train_test_split(trainDataX, trainDataY, test_size=0.1, random_state=0)
#clf = linear_model.ElasticNet(alpha = 0.1)
#fitted = clf.fit(X_train,y_train)
#y_predicted = clf.predict(X_test)
#print fitted.score(X_test, y_test)
###############################################################

# log data
trainDataY = np.log1p(trainDataY)

# transform and scale data
scaler = StandardScaler()
scaler.fit(trainDataX)
trainDataX = scaler.transform(trainDataX)
testDataX = scaler.transform(testDataX)

# train on the data
clf = linear_model.SGDRegressor(loss = "squared_loss", average=True)
fitted = clf.fit( trainDataX, trainDataY)
predicted = clf.predict(testDataX)

# reverse the log
predicted = (np.exp(predicted)).astype(int)

# create file
print('Generating submission file ...')
results = pd.DataFrame({'Demanda_uni_equil': predicted}, dtype=int)  
        
#Writting to csv
results.to_csv('regressor.csv', index=True, header=True, index_label='id')  