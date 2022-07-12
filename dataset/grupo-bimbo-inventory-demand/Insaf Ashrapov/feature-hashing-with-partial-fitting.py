import numpy as np
import pandas as pd
import math
import sklearn
import math
from sklearn.cross_validation import cross_val_score
from subprocess import check_output

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

from sklearn.linear_model import SGDRegressor

from sklearn.preprocessing import OneHotEncoder

 #   rmsle - error function used in LB
def rmsle_func(actual, predicted):
    return np.sqrt(msle(actual, predicted))
def msle(actual, predicted):
    return np.mean(sle(actual, predicted))
def sle(actual, predicted):
    return (np.power(np.log(np.array(actual)+1) - 
                np.log(np.array(predicted)+1), 2))

#  to decrease memory usage:          
dtypes = {'Semana' : 'int32',
                              'Agencia_ID' :'int32',
                              'Canal_ID' : 'int32',
                              'Ruta_SAK' : 'int32',
                              'Cliente-ID' : 'int32',
                              'Producto_ID':'int32',
                              'Venta_hoy':'float32',
                              'Venta_uni_hoy': 'int32',
                              'Dev_uni_proxima':'int32',
                              'Dev_proxima':'float32',
                              'Demanda_uni_equil':'int32'}

model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, 
                     fit_intercept=True, n_iter=10, shuffle=True, verbose=0, 
                     epsilon=0.1, learning_rate='invscaling', 
                     eta0=0.01, power_t=0.25, warm_start=True, average=False)

from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=8000, input_type = 'string') #8000 - the number of total unique values over all data


df_train = pd.read_csv('../input/train.csv', dtype  = dtypes, usecols=["Semana", "Agencia_ID", "Canal_ID", 'Ruta_SAK',
                                                             'Producto_ID','Demanda_uni_equil'], chunksize=90000)

i = 1
num = 40
#pd.concat([train, pd.get_dummies(train['Semana'],sparse=True)], axis=1, join_axes=[train.index])
for chunk in df_train:
    if  i < num :
        X_chunk = h.fit_transform(chunk[["Semana", "Agencia_ID", "Canal_ID", 'Ruta_SAK', 'Producto_ID']].astype('str').as_matrix())
        y_chunk = np.log(np.ravel(chunk[['Demanda_uni_equil']].as_matrix()) + 1)
        
        model.partial_fit(X_chunk, y_chunk)
        i = i + 1
    elif i == num:
        X_chunk = h.fit_transform(chunk[["Semana", "Agencia_ID", "Canal_ID", 'Ruta_SAK','Producto_ID']].astype('str').values)
        y_chunk = np.log(np.ravel(chunk[['Demanda_uni_equil']].values) + 1)
        
        #print ('rmsle: ', rmsle_func(y_chunk, model.predict(X_chunk)))
        print ('RMSE ', math.sqrt(sklearn.metrics.mean_squared_error(y_chunk, model.predict(X_chunk))))
        i = i + 1
    else:
         break
print ('Finished the fitting')

#predict

X_test = pd.read_csv('../input/test.csv',dtype  = dtypes,usecols=['id', "Semana", "Agencia_ID", "Canal_ID", 'Ruta_SAK',
                                                            'Producto_ID'])
ids = X_test['id']
X_test.drop(['id'], axis =1, inplace = True)

y_predicted = np.exp(model.predict(h.fit_transform(X_test.astype('str').values))) - 1

# to create post only non negative values
def nonnegative(x):
    if x > 0:
        return x
    else: 
        return 3.9


submission = pd.DataFrame({"id":ids, "Demanda_uni_equil": y_predicted})
#print (submission > 0).sum()
y_predicted = list(map(nonnegative, y_predicted))

submission = pd.DataFrame({"id":ids, "Demanda_uni_equil": y_predicted})
cols = ['id',"Demanda_uni_equil"]
submission = submission[cols]
submission.to_csv("submission.csv", index=False)


print('Completed!')