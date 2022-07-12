# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from catboost import CatBoostRegressor





train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_id=test.id


X = train.drop(['id','bandgap_energy_ev','formation_energy_ev_natom'], axis=1)
Y_feen = np.log(train['formation_energy_ev_natom']+1)
Y_bee = np.log(train['bandgap_energy_ev']+1)

test = test.drop(['id'], axis = 1)


def runCatBoost(x_train, y_train,x_test, y_test,test,depth):
    
    y_pred_train = 0
    y_pred_test = 0
    mse = 0
    
    for i in range(10):
        model=CatBoostRegressor(iterations=4000,
                                learning_rate=0.01,
                                depth=depth,
                                loss_function='RMSE',
                                eval_metric='RMSE',
                                random_seed=77*i+147,
                                od_type='Iter',
                                od_wait=50)
        model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True, verbose=False)
        y1 = model.predict(x_test)
        
        y_pred_train += 0.1*y1

        y2 = model.predict(test)
        y_pred_test += 0.1*y2
        
    mse = mean_squared_error(y_test, y_pred_train)
    
    return y_pred_train,mse,y_pred_test


kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=30)
for dev_index, val_index in kf.split(X):
    dev_X, val_X = X.loc[dev_index], X.loc[val_index]
    dev_y, val_y = Y_feen.loc[dev_index], Y_feen.loc[val_index]
    y_pred_feen,mse_feen,y_pred_test_feen=runCatBoost(dev_X, dev_y, val_X, val_y,test,depth=3)
print("Mean cv score : ", np.mean(mse_feen))



kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=30)
for dev_index, val_index in kf.split(X):
    dev_X, val_X = X.loc[dev_index], X.loc[val_index]
    dev_y, val_y = Y_bee.loc[dev_index], Y_bee.loc[val_index]
    y_pred_bee,mse_bee,y_pred_test_bee=runCatBoost(dev_X, dev_y, val_X, val_y,test,depth=4)
print("Mean cv score : ", np.mean(mse_bee))



sub=pd.DataFrame()
sub["id"]=test_id
sub["formation_energy_ev_natom"]=np.clip(np.exp(y_pred_test_feen)-1, 0, None)
sub["bandgap_energy_ev"]=np.exp(y_pred_test_bee)-1
mse_total=np.mean([mse_bee,mse_feen])
print(mse_total)
sub.to_csv(str(mse_total)+"_.csv",index=False)