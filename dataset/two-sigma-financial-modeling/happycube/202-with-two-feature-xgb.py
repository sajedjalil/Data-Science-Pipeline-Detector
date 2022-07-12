import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import math

import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

from operator import itemgetter

localrun = False
usepublic = False
vmode = False
all_features = False # use w/vmode for feature selection.  run twice, cutting # of rounds to peak round

# This is taken from Frans Slothoubers post on the contest discussion forum.
# https://www.kaggle.com/slothouber/two-sigma-financial-modeling/kagglegym-emulation

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2)*np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r

# From the xgboost script (along with the param settings)
    
# Function XGBOOST ########################################################
def xgb_obj_custom_r(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_mean = np.mean(y_true)
    y_median = np.median(y_true)
    c1 = y_true
    #c1 = y_true - y_mean
    #c1 = y_true - y_median
    grad = 2*(y_pred-y_true)/(c1**2)
    hess = 2/(c1**2)
    return grad, hess

def xgb_eval_custom_r(y_pred, dtrain):
    #y_pred = np.clip(y_pred, -0.075, .075)
#    y_pred[y_pred > .075] = .075
#    y_pred[y_pred < -.075] = -.075
    y_true = dtrain.get_label()
    ybar = np.sum(y_true)/len(y_true)
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - ybar)**2)
    r2 = 1 - ssres/sstot
    error = np.sign(r2) * np.absolute(r2)**0.5
    return 'error', error

env = kagglegym.make()
o = env.reset()

#excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
excl = ['id', 'sample', 'y', 'timestamp']
basecols = [c for c in o.train.columns if c not in excl]


rcol_orig = ['Dtechnical_20', 'y_prev_pred_avg_diff', 'Dtechnical_21', 'technical_43_prev', 'technical_20', 'y_prev_pred_avgT0', 'y_prev_pred_mavg5', 'y_prev_pred_avgT1', 'fundamental_8_prev', 'Dtechnical_40', 'technical_7_prev', 'technical_7', 'fundamental_5', 'Dtechnical_30', 'technical_32_prev', 'technical_14_prev', 'fundamental_1', 'fundamental_43_prev', 'Dfundamental_22', 'Dtechnical_35', 'Dtechnical_6', 'Dtechnical_17', 'Dtechnical_27', 'Dfundamental_42', 'fundamental_1_prev', 'Dtechnical_0', 'technical_40', 'technical_40_prev', 'fundamental_36', 'Dfundamental_33', 'Dfundamental_48', 'technical_27_prev', 'fundamental_62_prev', 'fundamental_41_prev', 'Dfundamental_50', 'fundamental_48', 'derived_2_prev', 'Dtechnical_18', 'fundamental_35', 'Dfundamental_49', 'fundamental_26_prev', 'technical_28_prev', 'Dfundamental_63', 'fundamental_10_prev', 'fundamental_36_prev', 'fundamental_16', 'Dfundamental_8', 'fundamental_32', 'fundamental_40_prev', 'derived_0', 'Dfundamental_32', 'fundamental_17', 'Dtechnical_7', 'fundamental_25', 'technical_35', 'Dtechnical_19', 'technical_35_prev', 'fundamental_8', 'Dtechnical_32', 'Dfundamental_18', 'Dtechnical_37', 'fundamental_33_prev', 'Dtechnical_28', 'fundamental_46', 'Dfundamental_1', 'Dfundamental_45', 'fundamental_18', 'technical_12', 'technical_44', 'fundamental_22', 'Dtechnical_5', 'technical_17_prev', 'Dfundamental_25']
rcol = rcol_orig.copy()

if all_features:
    rcol = []
    for c in basecols:
        rcol.append(c)
        rcol.append(c + '_prev')
        rcol.append('D' + c)

backy_fset = ['technical_13', 'technical_20', 'technical_13_prev', 'technical_20_prev', 'technical_30_prev', 'technical_30']
for f in backy_fset:
    if f not in rcol:
        rcol.append(f)

def get_basecols(rcol):
    duse = {}

    for r in rcol:
        if 'y' in r:
            continue

        if 'D' in r:
            duse[r[1:]] = True
        elif '_prev' in r:
            duse[r[:-5]] = True
        elif r in basecols:
            duse[r] = True

    return [k for k in duse.keys()]

basecols_touse = get_basecols(rcol)

if vmode:
    train = pd.read_hdf('../input/train.h5')
else:
    train = o.train.copy()

d_mean = o.train[basecols_touse].median(axis=0)
for c in basecols_touse:
    d_mean[c + '_prev'] = d_mean[c]
    d_mean['D' + c] = 0

median = {t[0]:t[1] for t in zip(d_mean.index, d_mean.values)}
median['y'] = 0

print('processed medians')

class DataPrep:
    
    def __init__(self, yprev_model = None, keepinput = True, cols = rcol):
        self.previnput = None
        self.prevavg = 0
        self.cols = cols.copy()
        
        self.basecols = get_basecols(self.cols)
        self.keepcols = ['y', 'id', 'timestamp'] + self.basecols
        
        self.allinput = [] if keepinput else None
        
        self.dayavg = []
        
        self.yprev_model = yprev_model
        
    def procday(self, day_in):
        
        if 'y' not in day_in and 'y' in self.keepcols:
            self.keepcols.remove('y')
        
        day = day_in[self.keepcols].copy()
        
        notinnew = []
        
        if self.previnput is not None:
            olen = len(day)
            day = pd.merge(day, self.previnput, on='id', how = 'left', suffixes=['', '_prev'])
            notinnew = self.previnput[~self.previnput.id.isin(day_in.id)].copy()
            #print(day.iloc[0].timestamp, len(notinnew))
        else:
            for c in self.basecols:
                day[c + '_prev'] = np.full_like(day[c], 0, dtype=np.float32)
                #day[c + '_prev'] = np.zeros_like(day[c], dtype=np.float32)
        
        for c in self.cols:
            if c == 'y_prev_pred':
                continue

            if c[0] == 'D':
                day[c] = day[c[1:]] - day[c[1:] + '_prev']
                
        self.previnput = day_in[self.keepcols].copy()
        if len(notinnew) > 0:
            self.previnput = self.previnput.append(notinnew[self.keepcols])
        
        if self.yprev_model:
            day['y_prev_pred'] = self.yprev_model.predict(day[backy_fset].fillna(d_mean).values.reshape(-1,len(backy_fset)))

            avg = day.y_prev_pred.mean()

            self.dayavg.append(avg)
            day['y_prev_pred_mavg5'] = np.ma.average(np.array(self.dayavg[-5:]))#, weights=range(1, len(self.dayavg[-10:]) + 1))
            day['y_prev_pred_min5'] = day.y_prev_pred - day.y_prev_pred_mavg5
            day['y_prev_pred_mavg5d'] = avg - np.ma.average(np.array(self.dayavg[-5:]))
            
            day['y_prev_pred_mavg10'] = np.ma.average(np.array(self.dayavg[-10:]))#, weights=range(1, len(self.dayavg[-10:]) + 1))
            day['y_prev_pred_mavg20'] = np.ma.average(np.array(self.dayavg[-20:]))
            day['y_prev_pred_mavg40'] = np.ma.average(np.array(self.dayavg[-40:]))
            
            day['y_prev_pred_avgT1'] = self.prevavg
            day['y_prev_pred_avgT0'] = avg
            day['y_prev_pred_avg_diff'] = avg - self.prevavg

            self.prevavg = avg
            
        if self.allinput is not None:
            self.allinput.append(day.copy())

        return day
    
    def run(self, df):
        assert self.allinput is not None
        
        for g in df.groupby('timestamp'):
            self.procday(g[1])
            
        return pd.concat(self.allinput)

yptrain = DataPrep(keepinput=True, cols=backy_fset).run(train)

#yptrain_prep = tmp.run(yptrain)

yptrain.sort_values(['id', 'timestamp'], inplace=True)

ypmodel = LinearRegression(n_jobs=-1)
low_y_cut = -0.0725
high_y_cut = 0.075

mask = np.logical_and(yptrain.y > low_y_cut, yptrain.y < high_y_cut)
for f in backy_fset:
    mask = np.logical_and(mask, ~yptrain[f].isnull())
    
yptraina = yptrain[mask]
ypmodel.fit(yptraina[backy_fset].values.reshape(-1,len(backy_fset)), yptraina.y_prev.fillna(0))

print(len(yptraina), ypmodel.coef_, ypmodel.intercept_)

#630210 [  4.94327753  10.22880098  -4.53049511  -9.34886941   8.94329596
#  -9.83007277] -2.68988841901e-05


d_mean['y'] = 0

start = time.time()

train = DataPrep(keepinput = True, yprev_model = ypmodel).run(train)

print('train proc')

endt = time.time()
print(endt - start)

dcol = [c for c in train.columns if c not in excl]

if usepublic:
    data_all = pd.read_hdf('../input/train.h5')

    #public = data_all[data_all.timestamp > 905]
    allpublic = DataPrep(yprev_model = ypmodel, keepinput=True).run(data_all)
    public = DataPrep(yprev_model = ypmodel, keepinput=True).run(data_all[data_all.timestamp > 905])

print(r_score(train.y_prev.fillna(0), train.y_prev_pred))

if usepublic:
    print(r_score(public.y_prev.fillna(0), public.y_prev_pred))

train.sort_values(['id', 'timestamp'], inplace=True)

print('prepping xgb now')
xtrain, xvalid = train_test_split(train, test_size = 0.2, random_state = 2017)

cols_to_use = [c for c in rcol if c in xtrain.columns and c in rcol_orig] 

                                                      
# Convert to XGB format
to_drop = ['timestamp', 'y']
xtrain = xtrain[np.abs(xtrain['y']) < 0.018976588919758796]
#xtrain = xtrain[np.abs(xtrain['y']) < 0.015]
train_xgb = xgb.DMatrix(data=xtrain[cols_to_use],
                        label=xtrain['y'])

#del xtrain

# determined by using testing w/public set that this matches public_xgb better.
# higher values make it end sooner.
#xvalid = xvalid[np.abs(xvalid['y']) > 0.009]
valid_xgb = xgb.DMatrix(data=xvalid[cols_to_use],
                        label=xvalid['y'])

#del xvalid

evallist = [(train_xgb, 'train'), (valid_xgb, 'valid')]

if usepublic:
    public_xgb = xgb.DMatrix(data=public[cols_to_use], label=public['y'])

    evallist = [(train_xgb, 'train'), (valid_xgb, 'xvalid'), (public_xgb, 'public')]

print('xtrain+valid')

params = {
    'objective': 'reg:linear'
    ,'eta': 0.08
    ,'max_depth': 3
    , 'subsample': 0.9
    #, 'colsample_bytree': 1
    ,'min_child_weight': 2**11
    #,'gamma': 100
    , 'seed': 10
    #, 'base_score': 0
}


model = []
for seed in [5041976, 31338]:
    params['seed'] = seed
    model.append(xgb.train(params.items()
                  , dtrain=train_xgb
                  , num_boost_round=90
                  , evals=evallist
                  , early_stopping_rounds=20
                  , maximize=True
                  , verbose_eval=10
                  , feval=xgb_eval_custom_r
                  ))

if not localrun:
    del train_xgb
    del valid_xgb
    if usepublic:
        del public_xgb

print('xgb done, linear now')

lin_features = ['Dtechnical_20', 'technical_20', 'Dtechnical_21']

def prep_linear(df, c = lin_features):
    df_tmp = df.fillna(d_mean)
    m2mat = np.zeros((len(df), len(c)))
    for i in range(len(c)):
        m2mat[:,i] = df_tmp[c[i]].values
    
    return m2mat

# Observed with histograns:
#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
traincut = train[np.logical_and(train.y > low_y_cut, train.y < high_y_cut)][['y'] + lin_features].copy().fillna(d_mean)

model2 = LinearRegression(n_jobs=-1)
model2.fit(prep_linear(traincut), traincut.y)

print('linear done')

if vmode:
    preds_xgb = model[0].predict(valid_xgb, ntree_limit=model[0].best_ntree_limit)
    preds_linear = model2.predict(prep_linear(xvalid))
    
    preds = (preds_xgb * 0.7) + (preds_linear * 0.3)
    #preds = preds_xgb
    
    rs = kagglegym.r_score(xvalid.y, preds)
    
    ID = 'expv-{0}.pkl'.format(int(rs * 10000000))
    print(rs, ID)
    
    #ID = 'subv-203172.pkl' # if actual submission
    
    output = xvalid[['id', 'timestamp', 'y']].copy()
    output['y_hat'] = preds
    output['y_hat_xgb'] = preds_xgb
    output['y_hat_linear'] = preds_linear
    
    output.to_pickle(ID)

if all_features:
    m = model[0]

    fs = m.get_fscore()
    fsl = [(f,fs[f]) for f in fs.keys()]
    fsl = sorted(fsl, key=itemgetter(1), reverse=True)

    len(fsl)

    print('rcol =', [f[0] for f in fsl])

razz_params = {
    'objective': 'reg:linear'
    ,'eta': 0.04
    ,'max_depth': 3
    , 'subsample': 0.9
    #, 'colsample_bytree': 1
    ,'min_child_weight': 2**11
    #,'gamma': 100
    , 'seed': 10
    #, 'base_score': 0
}

frazz = ['Dtechnical_20', 'technical_43_prev']
dtrain_razz = xgb.DMatrix(xtrain[frazz], label = xtrain.y)
dvalid_razz = xgb.DMatrix(xvalid[frazz], label = xvalid.y)
razz_evallist = [(dvalid_razz, 'xvalid'), (dtrain_razz, 'train')]
if usepublic:
    dpublic_razz = xgb.DMatrix(public[frazz], label = public.y)
    razz_evallist = [(dtrain_razz, 'train'), (dvalid_razz, 'xvalid'), (dpublic_razz, 'public')]
    
model_razz = xgb.train(razz_params.items()
                  , dtrain=dtrain_razz
                  , num_boost_round=400 if usepublic else 188
                  , evals=razz_evallist
                  , early_stopping_rounds=20
                  , maximize=True
                  , verbose_eval=10
                  , feval=xgb_eval_custom_r)

start = time.time()        

dprep = DataPrep(yprev_model = ypmodel, keepinput=localrun)
        

if localrun:
    env = kagglegym.make()
    o = env.reset()

while True:
    test_preproc = o.features.copy()
    
    #if c in basecols:
        #test_preproc.fillna(d_mean, inplace=True)
    
    test = dprep.procday(test_preproc)
    
    #test.fillna(0, inplace=True)
    
    test_xgb = xgb.DMatrix(data=test.drop(['id', 'timestamp'], axis=1)[cols_to_use])
    test_xgb_razz = xgb.DMatrix(data=test.drop(['id', 'timestamp'], axis=1)[frazz])

    xgbpreds = np.zeros(len(test), dtype=np.float64)
    for m in model:
        xgbpreds += m.predict(test_xgb, ntree_limit=m.best_ntree_limit)
    xgbpreds /= len(model)
    
    preds_razz = model_razz.predict(test_xgb_razz, ntree_limit=model_razz.best_ntree_limit)
    
    preds_linear = model2.predict(prep_linear(test)).clip(low_y_cut, high_y_cut)
    
    test_y = (preds_razz * 0.25) + (xgbpreds * 0.55) + (preds_linear * 0.2)
    
    o.target['y'] = test_y
    target = o.target
    
    timestamp = o.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{0} {1}".format(timestamp, time.time() - start))
        start = time.time()

    # We perform a "step" by making our prediction and getting back an updated "observation":
    o, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
    
    