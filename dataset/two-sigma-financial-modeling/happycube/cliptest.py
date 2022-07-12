import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

env = kagglegym.make()
o = env.reset()

#excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
excl = ['id', 'sample', 'y', 'timestamp']
col = [c for c in o.train.columns if c not in excl]

train = o.train.copy()

nlist = [c for c in col if train[c].isnull().sum() != 0]

d_mean = o.train[col].median(axis=0)

train['rtimestamp'] = train['timestamp'] % 100.0

train['znull'] = train.isnull().sum(axis=1)
for c in nlist:
    train[c + '_nan'] = train[c].isnull()
    
median = {c:train[c].median() for c in col}
    
#train = train.fillna(0)

cutlow = -.075
cuthigh = .075

def diffprev_largeset(df, firstseen):
    df2set = []

    for g in df.groupby('id'):
        gid = g[1].sort_values('timestamp')
        
        if g[0] not in firstseen:
            firstseen[g[0]] = gid.timestamp.min() if (gid.znull.max() == 106) else -1
        
        gid['elapsed'] = gid.timestamp - firstseen[g[0]] if (firstseen[g[0]] != -1) else -1

        for c in col:
            gid[c] = gid[c].fillna(median[c])
            
            d = np.zeros_like(gid[c])
            d[1:] = np.diff(gid[c].values)
            d[d != d] = 0

            p = np.zeros_like(gid[c])
            p[1:] = gid[c].values[0:-1]
            p[d != d] = 0

            gid['D' + c] = d
            gid[c + '_prev'] = p
            
        #gid = gid[np.logical_and(gid.y >= cutlow, gid.y < cuthigh)]
        #print(gid.y.min(), gid.y.max())

        df2set.append(gid)
        
    df2 = pd.concat(df2set)
    return df2

firstseen = {}

train = diffprev_largeset(train, firstseen)
#valid2 = diffprev_largeset(valid, firstseen)

dcol = [c for c in train.columns if c not in excl]

# 142, .001 - determined by modelx4 notebook run with all features (as above)
rcol = ['fundamental_7', 'fundamental_8', 'fundamental_11', 'fundamental_14', 'fundamental_17', 'fundamental_18', 'fundamental_21', 'fundamental_23', 'fundamental_40', 'fundamental_44', 'fundamental_45', 'fundamental_48', 'fundamental_50', 'fundamental_51', 'fundamental_52', 'fundamental_53', 'fundamental_54', 'fundamental_55', 'fundamental_56', 'fundamental_58', 'technical_2', 'technical_6', 'technical_7', 'technical_11', 'technical_12', 'technical_14', 'technical_17', 'technical_19', 'technical_20', 'technical_21', 'technical_22', 'technical_27', 'technical_29', 'technical_30', 'technical_31', 'technical_34', 'technical_35', 'technical_36', 'technical_40', 'technical_41', 'technical_43', 'rtimestamp', 'derived_1_nan', 'derived_3_nan', 'fundamental_17_nan', 'fundamental_25_nan', 'fundamental_27_nan', 'fundamental_41_nan', 'technical_0_nan', 'technical_5_nan', 'technical_9_nan', 'technical_12_nan', 'technical_16_nan', 'technical_18_nan', 'technical_24_nan', 'technical_25_nan', 'technical_31_nan', 'technical_32_nan', 'technical_37_nan', 'technical_38_nan', 'technical_39_nan', 'technical_41_nan', 'technical_44_nan', 'elapsed', 'Dderived_3', 'Dfundamental_0', 'fundamental_2_prev', 'fundamental_8_prev', 'Dfundamental_10', 'Dfundamental_11', 'fundamental_11_prev', 'Dfundamental_15', 'Dfundamental_17', 'Dfundamental_18', 'fundamental_18_prev', 'Dfundamental_20', 'fundamental_21_prev', 'Dfundamental_23', 'fundamental_23_prev', 'Dfundamental_30', 'Dfundamental_33', 'Dfundamental_36', 'Dfundamental_42', 'Dfundamental_45', 'fundamental_45_prev', 'fundamental_48_prev', 'fundamental_50_prev', 'Dfundamental_51', 'fundamental_52_prev', 'Dfundamental_53', 'fundamental_53_prev', 'Dfundamental_54', 'Dfundamental_56', 'Dfundamental_57', 'fundamental_58_prev', 'fundamental_59_prev', 'Dfundamental_60', 'Dtechnical_2', 'technical_2_prev', 'Dtechnical_3', 'Dtechnical_5', 'technical_5_prev', 'Dtechnical_6', 'technical_6_prev', 'Dtechnical_7', 'technical_7_prev', 'Dtechnical_11', 'technical_11_prev', 'Dtechnical_12', 'technical_12_prev', 'technical_13_prev', 'Dtechnical_14', 'technical_14_prev', 'technical_16_prev', 'Dtechnical_17', 'technical_17_prev', 'Dtechnical_19', 'technical_19_prev', 'Dtechnical_20', 'technical_20_prev', 'Dtechnical_21', 'technical_21_prev', 'technical_22_prev', 'Dtechnical_25', 'technical_25_prev', 'Dtechnical_27', 'technical_27_prev', 'Dtechnical_29', 'technical_29_prev', 'Dtechnical_30', 'technical_30_prev', 'Dtechnical_33', 'technical_33_prev', 'Dtechnical_35', 'technical_35_prev', 'Dtechnical_36', 'technical_36_prev', 'Dtechnical_37', 'Dtechnical_40', 'technical_40_prev', 'Dtechnical_43', 'technical_43_prev']

modelopts = [
#    [20, 5, 0],
    [100, 5, 17],
#    [20, 4, 0],
#    [20, 4, 17]
]

letr = []

for m in modelopts:
    letr.append(ExtraTreesRegressor(n_estimators=m[0], max_depth=m[1], n_jobs=-1, random_state=m[2], verbose=0))
    letr[-1].fit(train[rcol], train['y'])
    
del train

nancols = [c for c in rcol if '_nan' in c]
origcols = [c for c in col if c in rcol]

#env = kagglegym.make()
#o = env.reset()

prevtest = None

while True:
    test = o.features.copy()

    n = test.isnull().sum(axis=1)
    
    for c in nancols:
        test[c] = test[c[:-4]].isnull()
        
    for c in origcols:
        test[c] = test[c].fillna(median[c])
    
    test['znull'] = n
    test['rtimestamp'] = o.features['timestamp'] % 100.0
    
    # handle this in case we get multiple timestamps at once for a strange reason!
    for g in test.groupby('id'):
        if g[0] not in firstseen:
            firstseen[g[0]] = g[1].timestamp.min()
            
    test['elapsed'] = np.array([i[1].timestamp - firstseen[i[1].id] for i in test.iterrows()])
    
    if prevtest is not None:
        olen = len(test)
        test = pd.merge(test, prevtest, on='id', how = 'left', suffixes=['', '_prev'])
        #print(olen, len(test))
    else:
        for c in col:
            test[c + '_prev'] = np.zeros_like(test[c], dtype=np.float32)
        
    for c in rcol:
        if c[0] == 'D':
            test[c] = test[c[1:]] - test[c[1:] + '_prev']
            
        test[c] = test[c].fillna(0)
        
    prevtest = o.features.copy()
        
    pred = o.target

    lpreds = []
    for etr in letr:
        lpreds.append(etr.predict(test[rcol]))
    
    pred['y'] = np.mean(lpreds, axis=0)

    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(prevtest['timestamp'][0], reward, np.min(pred['y']), np.max(pred['y']))
