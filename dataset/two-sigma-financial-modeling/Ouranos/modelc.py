import kagglegym
import numpy as np
np.random.seed(43210)
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import math

    
def _reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return(R)

env = kagglegym.make()
o = env.reset()
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]

#train = pd.read_hdf('../input/train.h5')
print("Loading Data")
train=o.train

previousTechnical13 = {}
previousTechnical20 = {}
previousTechnical30 = {}

#d_mean= train.median(axis=0)

#train = o.train[col]
n = train.isnull().sum(axis=1)
train = train.fillna(0)
#train = train.fillna(d_mean)
print("Adding new Features")
train['znull'] = n
train['thmtw'] = train.technical_30-train.technical_20

listofcolumns=['technical_13','technical_20','technical_30']

def addcolumns(df,listofcolumns):
    for col in listofcolumns:
        df['prev'+col]=df.groupby('id')[col].shift(1).fillna(0)
        df['D'+col]  = df[col] - df['prev'+col]
    return(df)

train=addcolumns(train,listofcolumns)
train['Ddd'] = train.Dtechnical_20 - train.Dtechnical_30

def addcolumns2(df):
    df['Dtechnical_20_30']= df['Dtechnical_20'] * df['Dtechnical_30']
    df['Dtechnical_20_13']= df['Dtechnical_20'] * df['Dtechnical_13']
    df['Dtechnical_30_13']= df['Dtechnical_30'] * df['Dtechnical_13']
    df['Dtechnical_20m30_13']= (df['Dtechnical_20'] - df['Dtechnical_30']) * df['Dtechnical_13']
    return(df)

train=addcolumns2(train)

#target=o.train['y']
#mean_y=target.mean()
mean_train_y=(o.train['y']).mean()
print(mean_train_y)

colS = [c for c in train.columns if c not in excl]

print("Fitting XGBoost Model")
xgmat_train = xgb.DMatrix(train.loc[:,colS], label=train['y'])

params_xgb = {'objective': 'reg:linear',
              'eta'             : 0.2,
              'max_depth'       : 6,#4,
              'subsample'       : 0.9,
              'colsample_bytree':0.8,
              'min_child_weight': 10,
              'silent'          : 1,
              'seed'            : 22
              }

              
bst = xgb.train(params_xgb, xgmat_train, 35)

#XGBpredtrain=bst.predict(xgmat_train)
#XGBpredtrainMean=XGBpredtrain.mean()
#print(XGBpredtrainMean)

print("Fitting Extra Trees Model")
etr = ExtraTreesRegressor(n_estimators=50, max_depth=7, n_jobs=-1, random_state=17, verbose=0)
model1 = etr.fit(train[colS], train['y'])
train = []


#ymean_dict = dict(o.train.groupby(["id"])["y"].median())
reward_=[]
while True:
    test = o.features[col]
    firstsids = []
    yarray = np.zeros(o.target.y.shape[0])
    o.features.fillna(0, inplace=True)
    #o.features.fillna(median_values, inplace=True)
    timestamp = o.features["timestamp"][0]
    #llData = None
    allData = pd.DataFrame(np.zeros((test.shape[0], 6)))
    allrows=0
    for i in range(o.target.y.shape[0]):
        sid = o.features["id"].values[i]
        if(sid in previousTechnical20.keys()):
            data = np.zeros(shape=(1, 6))
            data[0, 0] = previousTechnical13[sid]
            data[0, 1] = o.features["technical_13"][i]
            data[0, 2] = previousTechnical20[sid]
            data[0, 3] = o.features["technical_20"][i]
            data[0, 4] = previousTechnical30[sid]
            data[0, 5] = o.features["technical_30"][i]

        else:
            data = np.zeros(shape=(1, 6))
            #data -= 999

        previousTechnical13[sid] = o.features["technical_13"][i]
        previousTechnical20[sid] = o.features["technical_20"][i]
        previousTechnical30[sid] = o.features["technical_30"][i]

        allData.loc[allrows,:] = data
        allrows += 1 
        
    allData.columns=['technical_13_Prev', 'technical_13_Cur', 'technical_20_Prev', 'technical_20_Cur',
                     'technical_30_Prev', 'technical_30_Cur']

    #for g in range(0,14):
    #    allData.iloc[:,g][allData.iloc[:,g]==-999] = 0#cmar[g]
    #allData.replace(-999,0,inplace=True)
    
    #test = o.features[col]
    n = test.isnull().sum(axis=1)
    #for c in test.columns:
    #    test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(0)
    #test = test.fillna(d_mean)
    test['znull'] = n
    test['thmtw'] = test.technical_30-test.technical_20
    
    test['prevtechnical_20']=allData['technical_20_Prev']
    test['Dtechnical_20']  = test['technical_20'] - test['prevtechnical_20']
    test['prevtechnical_30']=allData['technical_30_Prev']
    test['Dtechnical_30']  = test['technical_30'] - test['prevtechnical_30']
    test['prevtechnical_13']=allData['technical_13_Prev']
    test['Dtechnical_13']  = test['technical_13'] - test['prevtechnical_13']

    
    test['Ddd'] = test.Dtechnical_20 - test.Dtechnical_30
    test=addcolumns2(test)
    #test = test.fillna(0)
    #test = test[colS]
    ETpreds = model1.predict(test[colS])
    xgmat_test = xgb.DMatrix(test.loc[:,colS])
    XGBpreds=bst.predict(xgmat_test)
    XGBpreds=(XGBpreds - XGBpreds.mean()+mean_train_y).clip(-0.002,0.002)
    pred = o.target
    pred['y'] = (XGBpreds + ETpreds)/2
    #test2 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)
    #pred['y'] =predGP *0.6 + 0.4* (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.6) + (model2b.predict(test2).clip(low_y_cut, high_y_cut))*0.4 + (model2.predict(test2).clip(low_y_cut, high_y_cut) * 0.4)
    #pred['y'] =predGP *0.5 + 0.5* (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.6) +  (model2.predict(test2).clip(low_y_cut, high_y_cut) * 0.4)
    #pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    reward_.append(reward)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward, np.mean(np.array(reward_)))