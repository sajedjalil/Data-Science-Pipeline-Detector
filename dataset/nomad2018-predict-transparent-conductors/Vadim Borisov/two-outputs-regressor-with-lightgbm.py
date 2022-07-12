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

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from tqdm import tqdm


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_id=test.id


X = train.drop(['id','bandgap_energy_ev','formation_energy_ev_natom'], axis=1)
Y_feen = np.log(train['formation_energy_ev_natom']+1)
Y_bee = np.log(train['bandgap_energy_ev']+1)

test = test.drop(['id'], axis = 1)

X = X.fillna(X.mean())

two_Y = np.zeros((len(Y_bee),2))
two_Y[:,0] = Y_feen
two_Y[:,1] = Y_bee

y_pred = 0
N = 2
for i in tqdm(range(N)):
    model = MultiOutputRegressor(lgb.LGBMRegressor(random_state=i*101), n_jobs=-1)
    model.fit(X, two_Y)
    y_pred += model.predict(test)
y_pred /= N


sub=pd.DataFrame()
sub["id"]=test_id
sub["formation_energy_ev_natom"]=np.clip(np.exp(y_pred[:,0])-1, 0, None)
sub["bandgap_energy_ev"]=np.exp(y_pred[:,1])-1
sub.to_csv('sub.csv',index=False)