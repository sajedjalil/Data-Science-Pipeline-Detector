import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

#Loaded Data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
test_dataWithId = pd.read_csv("../input/test.csv")

#Training data
y = train_data[['Cover_Type']]
X = train_data.drop(['Cover_Type'], axis=1)
#Drop Id from Training and Test data
X = X.drop(['Id'], axis=1)
test_data = test_data.drop(['Id'], axis=1)

idx = 10 
cols = list(X.columns.values)[:idx]
X[cols] = StandardScaler().fit_transform(X[cols])
test_data[cols] = StandardScaler().fit_transform(test_data[cols])

# svm_parameters = [{'kernel': ['rbf'], 'C': [1,10,100,1000]}]
# model = GridSearchCV(SVC(), svm_parameters, cv=3, verbose=2)

# model.fit(X, y.iloc[:,0])    
# print(model.best_params_)
# print(model.cv_results_)

model = SVC(C=1000, kernel='rbf')
model.fit(X, y.iloc[:,0])    
print(model.score(X, y.iloc[:,0]))
predictions = model.predict(test_data)
print(predictions)
c1 = pd.DataFrame(test_dataWithId["Id"])
c2 = pd.DataFrame({'Cover_Type': predictions})
res = (pd.concat([c1, c2], axis=1))
res.to_csv('ouput.csv', index=False)