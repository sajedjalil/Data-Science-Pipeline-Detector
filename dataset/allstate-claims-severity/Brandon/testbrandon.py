import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
 
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

data = pd.read_csv('../input/train.csv')
print('load in data: train.csv')

data.drop('id',axis=1,inplace=True)
print(data.shape)

#print all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


#print "Feature columns:\n{}".format(feature_cols)
#print "\nTarget column: {}".format(target_cols)
#print "feature columns ({} total features):\n".format(len(feature_cols))
#print(data.head())

#row, col = data.shape



print('normalizing target label value')
#target_cols = data.columns[-1]

data['loss'] = np.log1p(data['loss'])
# print('getting labels')
target_data = data['loss']
#print(target_data.shape)
#print(target_data.head())
# cat_col = 116
# labels= []

# for i in range(0,cat_col):
#     labels.append(feature_cols[i])
    
#print(labels)
print('encoding categorical data')
encoded_df = encode_onehot(data, cols=data.columns)

del data
#print(encoded_df.shape)
#print(encoded_df.head())

feature_cols = list(encoded_df.columns[0:-1])
target_cols = encoded_df.columns[-1]
X_all = encoded_df[feature_cols]
y_all = encoded_df['loss']

print(encoded_df.shape)
#print(y_all.head(1))
del encoded_df

#print(X_all.head())
print("split training data")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=0)

print('train model')
# from sklearn.neighbors import KNeighborsRegressor
# n_neighbors = 1
# model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)

# from sklearn.linear_model import LinearRegression
# model = LinearRegression(n_jobs=-1)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5,random_state=0)


model.fit(X_train, y_train)

# print(X_train.shape)
print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

print('make prediction')
from sklearn.metrics import mean_absolute_error
predict = model.predict(X_test)
result = mean_absolute_error(np.expm1(y_test), np.expm1(predict))

print(result)

#print(predict.shape)


# print("calculating mae")
# #Accuracy of the model using all features
# result = mean_absolute_error(np.expm1(y_value), np.expm1(model.predict(X_value)))
# print(result)
