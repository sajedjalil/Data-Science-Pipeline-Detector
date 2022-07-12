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
from sklearn.decomposition import PCA

def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_id=test.id

#train = train.drop(['id'], axis=1)
#train = train.drop_duplicates()

#train = train.reset_index(drop=True)
vector = np.vstack((train[['lattice_vector_1_ang', 'lattice_vector_2_ang','lattice_vector_3_ang']].values,
                    test[['lattice_vector_1_ang', 'lattice_vector_2_ang','lattice_vector_3_ang']].values))

pca = PCA().fit(vector)
train['vector_pca0'] = pca.transform(train[['lattice_vector_1_ang', 'lattice_vector_2_ang','lattice_vector_3_ang']])[:, 0]
test['vector_pca0'] = pca.transform(test[['lattice_vector_1_ang', 'lattice_vector_2_ang','lattice_vector_3_ang']])[:, 0]


X = train.drop(['id','bandgap_energy_ev','formation_energy_ev_natom'], axis=1)
Y_feen = np.log(train['formation_energy_ev_natom']+1)
Y_bee = np.log(train['bandgap_energy_ev']+1)

test = test.drop(['id'], axis = 1)


def runCatBoost(x_train, y_train,x_test, y_test,test,depth):
    model=CatBoostRegressor(iterations=1200,
                            learning_rate=0.03,
                            depth=depth,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=99,
                            od_type='Iter',
                            od_wait=50)
    model.fit(x_train, y_train, eval_set=(x_test, y_test), use_best_model=True, verbose=False)
    y_pred_train=model.predict(x_test)
    rmsle_result = rmsle(np.exp(y_pred_train)-1,np.exp(y_test)-1)
    y_pred_test=model.predict(test)
    return y_pred_train,rmsle_result,y_pred_test

pred_full_test_cat_feen = 0
mse_cat_list_feen=[]
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=30)
for dev_index, val_index in kf.split(X):
    dev_X, val_X = X.loc[dev_index], X.loc[val_index]
    dev_y, val_y = Y_feen.loc[dev_index], Y_feen.loc[val_index]
    y_pred_feen,rmsle_feen,y_pred_test_feen=runCatBoost(dev_X, dev_y, val_X, val_y,test,depth=4)
    mse_cat_list_feen.append(rmsle_feen)
    pred_full_test_cat_feen = pred_full_test_cat_feen + y_pred_test_feen
mse_cat_feen_mean=np.mean(mse_cat_list_feen)
print("Mean cv score : ", np.mean(mse_cat_feen_mean))
y_pred_test_feen=pred_full_test_cat_feen/10

pred_full_test_cat_bee = 0
mse_cat_list_bee=[]
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=30)
for dev_index, val_index in kf.split(X):
    dev_X, val_X = X.loc[dev_index], X.loc[val_index]
    dev_y, val_y = Y_bee.loc[dev_index], Y_bee.loc[val_index]
    y_pred_bee,rmsle_bee,y_pred_test_bee=runCatBoost(dev_X, dev_y, val_X, val_y,test,depth=4)
    mse_cat_list_bee.append(rmsle_bee)
    pred_full_test_cat_bee = pred_full_test_cat_bee + y_pred_test_bee
mse_cat_bee_mean=np.mean(mse_cat_list_bee)
print("Mean cv score : ", np.mean(mse_cat_bee_mean))
y_pred_test_bee=pred_full_test_cat_bee/10


sub=pd.DataFrame()
sub["id"]=test_id
sub["formation_energy_ev_natom"]=np.clip(np.exp(y_pred_test_feen)-1, 0, None)
sub["bandgap_energy_ev"]=np.clip(np.exp(y_pred_test_bee)-1, 0, None)
rmsle_total=np.mean([mse_cat_bee_mean,mse_cat_feen_mean])
print(rmsle_total)
sub.to_csv(str(rmsle_total)+"_.csv",index=False)