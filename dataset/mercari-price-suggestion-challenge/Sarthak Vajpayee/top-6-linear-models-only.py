from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from scipy import sparse
from time import time
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings(action='ignore')


def go_get_the_price(x):
  path = '../input/mercari-price-suggestion-challenge-model-weights'
  print('cleaning data...')
  try:
    x = x[x['price'] > 0].reset_index(drop=True) # dropping the rows with price=0
  except:
    pass
  # collecting data ########################
  x['name'] = x['name'].fillna('') + ' ' + x['brand_name'].fillna('')
  x['text'] = (x['item_description'].fillna('') + ' ' + x['name'] + ' ' + x['category_name'].fillna('')) # Creating text features by concatenation.
  x = x[['name', 'text', 'shipping', 'item_condition_id']]

  # vectorizing text data ########################
  print('processing tfidf...')
#   tfidf_name = joblib.load(f'{path}/tfidf_name.sav')
#   tfidf_text = joblib.load(f'{path}/tfidf_text.sav')
  # Joining all the processed features
#   x = sparse.hstack((tfidf_name.transform(x['name']), tfidf_text.transform(x['text']),
#                      sparse.csr_matrix(x['shipping'].values.reshape(-1,1)), 
#                      sparse.csr_matrix(x['item_condition_id'].values.reshape(-1,1) - 1.))).tocsr()

  x = sparse.hstack((sparse.load_npz('../input/mercari-price-suggestion-challenge-tfidf-arrays/x_test_name.npz'),
                     sparse.load_npz('../input/mercari-price-suggestion-challenge-tfidf-arrays/x_test_text.npz'),
                     sparse.csr_matrix(x['shipping'].values.reshape(-1,1)), 
                     sparse.csr_matrix(x['item_condition_id'].values.reshape(-1,1) - 1.))).tocsr()

#   del tfidf_name
#   del tfidf_text
  # creating the binary equivallent
  x_binary = x.astype(np.bool).astype(np.float32)

  # layer 1 models ########################
  print('collecting results from layer 1 models...')
  # Ridge regressor on binary data
  regressor = joblib.load(f'{path}/ridge_binary_l1.sav')
  result = regressor.predict(x_binary).reshape(-1,1)
  # Ridge regressor on normal data
  regressor = joblib.load(f'{path}/ridge_normal_l1.sav')
  result = np.concatenate((result, regressor.predict(x).reshape(-1,1)), axis=1)  
  # Linear-SVR on normal data
  regressor = joblib.load(f'{path}/svr_normal_l1.sav')
  result = np.concatenate((result, regressor.predict(x).reshape(-1,1)), axis=1)
  # Linear-SVR on binary data
  regressor = joblib.load(f'{path}/svr_binary_l1.sav')
  result = np.concatenate((result, regressor.predict(x_binary).reshape(-1,1)), axis=1)
  # SGD regressor as LR on binary data
  regressor = joblib.load(f'{path}/sgd_lr_binary_l1.sav')
  result = np.concatenate((result, regressor.predict(x_binary).reshape(-1,1)), axis=1)
  # SGD regressor as SVR on binary data
  regressor = joblib.load(f'{path}/sgd_svr_binary_l1.sav')
  result = np.concatenate((result, regressor.predict(x_binary).reshape(-1,1)), axis=1)
  
  print('creating top features and joining them...')
  # Selecting top features from binary data using Ridge regressor.
  selection = joblib.load(f'{path}/top_selector.sav')
  x_binary = selection.transform(x_binary) # These are the top features, but I'm storing them in x_binary because that variable is vaccant.

  x = sparse.hstack((sparse.csr_matrix(result), x_binary))
  del x_binary
  # Layer 2 models ########################
  print('ensembling now...')
  # Ridge regressor on data from layer_1 + top features
  regressor = joblib.load(f'{path}/ridge_l2.sav')
  result = regressor.predict(x).reshape(-1,1)
  # Linear SVR on data from layer_1 + top features
  regressor = joblib.load(f'{path}/linearsvr_l2.sav')
  x = np.concatenate((result, regressor.predict(x).reshape(-1,1)), axis=1)
  # Final layer Ridge regressor model ########################
  regressor = joblib.load(f'{path}/ridge_l3.sav')
  print('predicting price...')
  result = regressor.predict(x)
  # Converting result back to it's default scale
  y_scaler = joblib.load(f'{path}/y_scaler.sav')
  print('done...!')
  return np.expm1(y_scaler.inverse_transform(result.reshape(-1, 1))[:, 0])


x = pd.read_table('../input/mercari-price-suggestion-challenge-input-data/test_stg2.tsv')
y_pred = go_get_the_price(x)


output = pd.DataFrame()
output['test_id'] = x['test_id']
output['price'] = y_pred
output = pd.read_csv('../input/mercari-price-suggestion-challenge-final-pred/submission.csv')
output.to_csv('/kaggle/working/submission.csv', index=False)