import pandas as pd
import numpy as np

data = pd.read_csv('../input/train.csv')
print('load in data: train.csv')

data.drop('id',axis=1,inplace=True)
#print(data.shape)

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# print(data.head(1))

print('normalizing target value')
data['loss'] = np.log1p(data['loss'])


#prepare data for training
#apply onehot encoding to data
print('encoding categorical data')
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

encoded_df = encode_onehot(data, cols=data.columns)

del data
# print(encoded_df.shape)
feature_cols = list(encoded_df.columns[0:-1])
target_cols = encoded_df.columns[-1]
X_all = encoded_df[feature_cols]
y_all = encoded_df['loss']

del encoded_df

print("split training data")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=0)

del X_all
del y_all

from time import time
def train_model(model, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    start = time()
    #from sklearn.cross_validation import ShuffleSplit
    #cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
    
    #from sklearn.grid_search import GridSearchCV
    #gammas = np.logspace(-6, -1, 10)
    #classifier = GridSearchCV(estimator=clf, cv=cv, param_grid=dict(gamma=gammas))
    #classifier.fit(X_train, y_train)
    # Start the clock, train the classifier, then stop the clock
    model.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start)) 
           

print('train model')
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5,random_state=0)
train_model(model, X_train, y_train)

print('make prediction')
from sklearn.metrics import mean_absolute_error
predict = model.predict(X_test)
result = mean_absolute_error(np.expm1(y_test), np.expm1(predict))

print(result)