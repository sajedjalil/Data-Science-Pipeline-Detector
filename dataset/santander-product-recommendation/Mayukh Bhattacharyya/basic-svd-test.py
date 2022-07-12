import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.linalg import sqrtm
from copy import deepcopy


usecols1 = ['fecha_dato', 'ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

train = pd.read_csv("../input/train_ver2.csv", usecols=usecols1)
train1 = train[train['fecha_dato']=="2016-05-28"].drop("fecha_dato", axis = 1)
train2 = train[train['fecha_dato']=="2016-04-28"].drop("fecha_dato", axis = 1)
#train3 = train[train['fecha_dato']=="2016-03-28"].drop("fecha_dato", axis = 1)

train1.to_csv('train1.csv', index=False)
train2.to_csv('train2.csv', index=False)

