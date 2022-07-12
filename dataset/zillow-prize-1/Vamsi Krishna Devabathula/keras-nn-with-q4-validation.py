
# Skeleton Based on Prasun Mishra's notebook
#   https://www.kaggle.com/prasunmishra/ann-using-keras

# Revised by Andy Harless 9/16 - 9/20
#   (added Dropout, BatchNormalization, wider layers, larger batches, more epochs)


# Parameters

INCLUDE_TIME_FEATURES = True
BASELINE_FUDGE_FACTOR = 1.033  # Optimal in XGB CV
FUDGE_FACTOR_SCALEDOWN = .25
N_EPOCHS = 150
BEST_EPOCH = False
CV_ONLY = False



from datetime import datetime
import numpy as np
import numpy as numpy
import pandas as pd
import pylab
import calendar
from scipy import stats
import seaborn as sns
from sklearn import model_selection, preprocessing
from scipy.stats import kendalltau
import warnings
import matplotlib.pyplot as plt
import pandas
## Keras comes here
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras import callbacks
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg




#################
##  READ DATA  ##
#################

# Load train, Prop and sample
print('Loading train, prop and sample data')
train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')

print('Fitting Label Encoder on properties')
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))
        
#Create df_train and x_train y_train from that
print('Creating training set:')
df_train = train.merge(prop, how='left', on='parcelid')

# Create df_test and test set
print('Creating df_test  :')
sample['parcelid'] = sample['ParcelId']

print("Merge Sample with property data :")
df_test = sample.merge(prop, on='parcelid', how='left')




###################################
##  CREATE TRAIN, CV, TEST SETS  ##
###################################

###########################################################
df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
if INCLUDE_TIME_FEATURES:
    df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
    df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
    df_train["transactiondate"] = df_train["transactiondate"].dt.day
    
select_qtr4 = df_train["transactiondate_quarter"] == 4

#df_train =df_train[ df_train.logerror > -0.4005 ]
#df_train=df_train[ df_train.logerror < 0.412 ]

print('Create x_train and y_train from df_train' )
x_train_all = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode',
                             'fireplacecnt', 'fireplaceflag'], axis=1)
if not INCLUDE_TIME_FEATURES:
     x_train_all = x_train_all.drop(['transactiondate_quarter'], axis=1)

y_train_all = df_train["logerror"]
y_train = y_train_all[~select_qtr4]
x_train = x_train_all[~select_qtr4]
x_valid = x_train_all[select_qtr4]
y_valid = y_train_all[select_qtr4]

y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

if INCLUDE_TIME_FEATURES:
    df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # typical date for 2016 test data
    df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
    df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
    df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
    df_test["transactiondate"] = df_test["transactiondate"].dt.day 
    
#################################

x_test = df_test[train_columns]

print('Shape of x_test:', x_test.shape)
print("Preparing x_test:")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
  



###################
##  PREPROCESS  ##
###################

#############################Imputer##################

from sklearn.preprocessing import Imputer
imputer= Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
imputer.fit(x_valid.iloc[:, :])
x_valid = imputer.transform(x_valid.iloc[:, :])
imputer.fit(x_test.iloc[:, :])
x_test = imputer.transform(x_test.iloc[:, :])

#########################Standard Scalar##############

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_valid = sc.fit_transform(x_valid)
x_val = np.array(x_valid)
y_val = np.array(y_valid)

################################################




#####################
##  TRAIN NETWORK  ##
#####################

len_x=int(x_train.shape[1])
print("len_x is:",len_x)

nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))

nn.compile(loss='mae', optimizer=Adam(lr=5e-3,decay=1e-4))

wtpath = 'weights.hdf5'
bestepoch = ModelCheckpoint( filepath=wtpath, verbose=1, save_best_only=True )

nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = N_EPOCHS, verbose=2, 
       validation_data=(x_val,y_val), callbacks=[bestepoch])

if BEST_EPOCH:       
    nn.load_weights( wtpath )

valid_pred = nn.predict(x_val)




#############################
##  OPTIMIZE FUDGE FACTOR  ##
#############################

print("\nMean absolute validation error with baseline fudge factor: ")
print(mean_absolute_error(y_valid, BASELINE_FUDGE_FACTOR*valid_pred))

mod = QuantReg(y_valid, valid_pred)
res = mod.fit(q=.5)
print("\nLAD Fit for Fudge Factor:")
print(res.summary())
fudge = res.params[0]
print("Optimized fudge factor:", fudge)

print("\nMean absolute validation error with optimized fudge factor: ")
print(mean_absolute_error(y_valid, fudge*valid_pred))

fudge **= FUDGE_FACTOR_SCALEDOWN

print("Scaled down fudge factor:", fudge)

print("\nMean absolute validation error with scaled down fudge factor: ")
print(mean_absolute_error(y_valid, fudge*valid_pred))

print("\nx_test.shape:",x_test.shape)




#################################
##  PREDICT AND WRITE RESULTS  ##
#################################

if not CV_ONLY:
    print("\nGenerating predictions...")
    y_pred_ann = fudge*nn.predict(x_test)

    print( "\nPreparing results for write..." )
    y_pred = y_pred_ann.flatten()

    output = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32),
            '201610': y_pred, '201611': y_pred, '201612': y_pred,
            '201710': y_pred, '201711': y_pred, '201712': y_pred})
    # set col 'ParceID' to first col
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]

    print( "\nWriting results to disk:" )
    output.to_csv('Only_ANN_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)



print( "\nFinished!" )


# version 19: Remove Nikunj features (maybe try later with retuning?)
# version 21: Optimize fudge factor
# version 22: Scale down optimized fudge factor
# version 23: Use best epoch
# version 25: 150 epochs, fudge_factor_scaledown .65->.5
# version 26: Make "best epoch" optional
# version 29: 150 epochs, fudge_factor_scaledown .5->.35
# version 30: Add CV_ONLY option, some code cleanup, fudge_factor_scaledown .35->.25