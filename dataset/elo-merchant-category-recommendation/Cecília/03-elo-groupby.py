import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from scipy.stats import skew 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
# models
from xgboost import XGBRegressor
import warnings

# Ignore useless warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Avoid runtime error messages
pd.set_option('display.float_format', lambda x:'%f'%x)

# make notebook's output stable across runs
np.random.seed(42)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import os
print(os.listdir("../input"))

all_new_hist = pd.read_csv("../input/all-new-hist/all_new_hist.csv")
all_new_hist.shape
#(28558483, 14)

newTransGr = pd.DataFrame()

listGroup = all_new_hist.groupby(['card_id'])['card_id'].count()
newTransGr['card_id'] = listGroup.index

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['city_id'].agg(lambda x: x.value_counts().index[0]))
newTransGr['City_Mode'] = listGroup['city_id']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['category_1'].agg(lambda x: x.value_counts().index[0]))
newTransGr['Cat1_Mode'] = listGroup['category_1']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['installments'].agg(lambda x: x.value_counts().index[0]))
newTransGr['Install_Mode'] = listGroup['installments']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['category_3'].agg(lambda x: x.value_counts().index[0]))
newTransGr['Cat3_Mode'] = listGroup['category_3']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['merchant_category_id'].agg(lambda x: x.value_counts().index[0]))
newTransGr['Merch_Mode'] = listGroup['merchant_category_id']

listGroup = pd.DataFrame(all_new_hist.groupby(['card_id'], as_index=False)['month_lag'].mean())
newTransGr['Mon_mean'] = listGroup['month_lag']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['purchase_amount'].sum())
newTransGr['purchase_amount'] = listGroup['purchase_amount']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['category_2'].agg(lambda x: x.value_counts().index[0]))
newTransGr['Cat2_Mode'] = listGroup['category_2']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['state_id'].agg(lambda x: x.value_counts().index[0]))
newTransGr['State_Mode'] = listGroup['state_id']

listGroup = pd.DataFrame(all_new_hist.groupby('card_id', as_index=False)['subsector_id'].agg(lambda x: x.value_counts().index[0]))
newTransGr['Subsec_Mode'] = listGroup['subsector_id']

newTransGr.to_csv("newTransGr2.csv", index=False)


