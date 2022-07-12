# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Census dataset
data = pd.read_csv("../input/train_V2.csv")
pred_data = pd.read_csv('../input/test_V2.csv')
cat_data = data['matchType']
cat_pred_data = pred_data['matchType']
data = data.drop(columns=['matchType'], axis=1)
pred_data = pred_data.drop(columns=['matchType'], axis=1)

X = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
X = scaler.fit_transform(X)

features_final = pd.get_dummies(cat_data)
features = features_final.iloc[:,:-1].values 

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

final_data = np.array(np.concatenate((X, features), 1))

y = np.array(y).reshape(-1,1)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
imputer = imputer.fit(y)
y = imputer.transform(y)

y = np.array(y).reshape(1,-1)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(final_data, y)

#temp = np.where(np.isnan(y))
#print(temp)


#y_pred = regressor.predict(final_data)
#print(y_pred)


#Now Predict data

X_predict = pred_data.iloc[:,3:]
Id = pred_data['Id']

Id = np.array(Id).reshape(-1,1)

X_predict = scaler.fit_transform(X_predict)
features_final_1 = pd.get_dummies(cat_pred_data)
features_1 = features_final_1.iloc[:,:-1].values 

X_predict = pca.fit_transform(X_predict)

final_data_1 = np.array(np.concatenate((X_predict, features_1), 1))

y_predict = regressor.predict(final_data_1)

y_predict = np.array(y_predict).reshape(-1,1)

output = np.array(np.concatenate((Id, y_predict), 1))

output = pd.DataFrame(output,columns = ["Id","winPlacePerc"])

output.to_csv('out.csv',index = False)




