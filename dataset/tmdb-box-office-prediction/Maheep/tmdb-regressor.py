import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

#load training data
train_data=pd.read_csv("../input/train.csv")

x_popularity = train_data['popularity'].mean()
train_data['popularity'] = np.where(train_data['popularity'].isnull(), x_popularity , train_data['popularity'])

x_budget = train_data['budget'].mean()
train_data['budget'] = np.where( (train_data['budget'].isnull()) | (train_data['budget'] == 0), x_budget , train_data['budget'])

x_runtime = train_data['runtime'].mean()
train_data['runtime'] = np.where(train_data['runtime'].isnull(), x_runtime , train_data['runtime'])

x_original_language = train_data['original_language'].mode()
train_data['original_language'] = np.where(train_data['original_language'].isnull(), x_original_language , train_data['original_language'])

#create a LabelEncoder object and fit it to each feature which contain textual data

number=LabelEncoder()
train_data['original_language']=number.fit_transform(train_data["original_language"].astype('str'))

#convert popularity and runtime into integers

train_data['popularity']=train_data["popularity"].astype(dtype=np.int64)
train_data['runtime']=train_data["runtime"].astype(dtype=np.int64)

#considering only specific feature
X_train=train_data[['popularity','budget','runtime','original_language']]
Y_train=train_data['revenue']



#load test data
test_data=pd.read_csv("../input/test.csv")

x_popularity = test_data['popularity'].mean()
test_data['popularity'] = np.where(test_data['popularity'].isnull(), x_popularity , test_data['popularity'])

x_budget = test_data['budget'].mean()
test_data['budget'] = np.where(test_data['budget'].isnull() | (train_data['budget'] == 0), x_budget , test_data['budget'])

x_runtime = test_data['runtime'].mean()
test_data['runtime'] = np.where(test_data['runtime'].isnull(), x_runtime , test_data['runtime'])

x_original_language =test_data['original_language'].mode()
test_data['original_language'] = np.where(test_data['original_language'].isnull(), x_original_language , test_data['original_language'])

#create a LabelEncoder object and fit it to each feature which contain textual data

number=LabelEncoder()
test_data['original_language']=number.fit_transform(test_data["original_language"].astype('str'))

#convert popularity and runtime into integers

test_data['popularity']=test_data["popularity"].astype(dtype=np.int64)
test_data['runtime']=test_data["runtime"].astype(dtype=np.int64)

#considering only specific feature
X_test=test_data[['popularity','budget','runtime','original_language']]

# Create GradientBoostingRegressor object 
model = GradientBoostingRegressor(loss='huber',n_estimators=150)
#fit training data into model
model.fit(X_train, Y_train)

#Predict Output
predicted= model.predict(X_test)
predicted=abs(predicted)


#For creating submission file
test_data=pd.read_csv("../input/test.csv")

#create submission file
submission = np.empty((4398,2),dtype=int)
submission[:,0] = test_data["id"]
submission[:,1] = predicted
submission = pd.DataFrame(data=submission,columns=["id","revenue"])
submission.to_csv("submission.csv",index = False)