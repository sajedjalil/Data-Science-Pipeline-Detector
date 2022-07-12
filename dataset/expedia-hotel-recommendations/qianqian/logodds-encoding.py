import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
from sklearn.base import BaseEstimator

class BatchLogoddsFeature(BaseEstimator):
    def __init__(self,yname,features,MIN_CAT_COUNTS=1):
        self.yname = yname
        assert type(features) == list
        self.features = features
        self.feature_dicts = dict()
        self.MIN_CAT_COUNTS = MIN_CAT_COUNTS


    def merge_res(self,x):
        x_temp = x[0]
        for i in range(1,len(x)):
            x_temp = pd.merge(x_temp,x[i], how='outer', left_index=True, right_index=True)
        return x_temp

    def fit(self,X,y=None):
        yname = self.yname
        features = self.features
        assert len(features)>0
        if len(features)>1:
            cname = 'x'.join(features)
        else:
            cname = features[0]
        self.cname = cname

        t_counts = []
        c_t_counts = []
        c_counts = []
        count = 0
        total_data = 0
        columns = []
        targets = []
        for temp in X: 
            if len(features)>1:
                temp[features] =  temp[features].astype(str)
                temp[cname] = temp[features].sum(axis=1)

            data = temp[[cname,yname]].astype(str)
            columns.append(data[cname].values)
            targets.append(data[yname].values)

            total_data+=data.shape[0]
            t_count=data.groupby([yname]).size()
            c_t_count=data.groupby([cname,yname]).size()
            c_count=data.groupby([cname]).size()

            t_count = pd.DataFrame(t_count)
            t_count.columns = ['cnt%s'%count]

            c_t_count = pd.DataFrame(c_t_count)
            c_t_count.columns = ['cnt%s'%count]

            c_count = pd.DataFrame(c_count)
            c_count.columns = ['cnt%s'%count]

            t_counts.append(t_count)
            c_t_counts.append(c_t_count)
            c_counts.append(c_count)
            count +=1

        columns=np.concatenate(columns)
        targets=np.concatenate(targets)
        columns = sorted(np.unique(columns))
        targets = sorted(np.unique(targets))

        t_counts = self.merge_res(t_counts)
        t_counts = t_counts.fillna(0).sum(axis=1)

        c_t_counts = self.merge_res(c_t_counts)
        c_t_counts = c_t_counts.fillna(0).sum(axis=1)

        c_counts = self.merge_res(c_counts)
        c_counts = c_counts.fillna(0).sum(axis=1)

        logodds={}
        MIN_CAT_COUNTS=self.MIN_CAT_COUNTS
        default_logodds=np.log(t_counts/float(total_data))-np.log(1.0-t_counts/float(total_data))
        logodds['default']=deepcopy(default_logodds)
        for col in columns:
            PA=c_counts[col]/float(total_data)
            logodds[col]=deepcopy(default_logodds)
            for cat in c_t_counts[col].keys():
                if (c_t_counts[col][cat]>MIN_CAT_COUNTS) and c_t_counts[col][cat]<c_counts[col]:
                    PA=c_t_counts[col][cat]/float(c_counts[col])
                    logodds[col][targets.index(cat)]=np.log(PA)-np.log(1.0-PA)


            logodds[col]=pd.Series(logodds[col])
            logodds[col].index=range(len(targets))

        self.logodds=logodds
            

    def transform(self,X):
        X = X.astype(str)
        features = self.features
        res = []
        cname = self.cname
        if len(features)>1:
            X[cname] = X[features].sum(axis=1)
        
        logodds= self.logodds
        new_features = X[cname].apply(lambda x: logodds[x] if x in logodds else logodds['default'])
        new_features.columns=["logodds_"+cname+"_"+str(x) for x in range(len(new_features.columns))]
        return new_features


    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X)

def sort_res(y_pred):
    res = []
    for yp in y_pred:
        yp = 1-yp
        temp = np.argsort(yp)[:5]
        res.append(temp.tolist())
    return res

train = pd.read_csv('../input/train.csv',parse_dates=['date_time'],iterator=True,chunksize=128000)
test = pd.read_csv('../input/test.csv',parse_dates=['date_time'],iterator=True,chunksize=128000)

blf = BatchLogoddsFeature(yname='hotel_cluster',features=['user_location_city', 'orig_destination_distance'],MIN_CAT_COUNTS=1)
blf.fit(train)
res = []
for chunk in test:
    y_pred = blf.transform(chunk).values
    res.append(y_pred)

res = np.hstack(res)
res = sort_res(res)
sub = []

for r in res:
    s = '%s %s %s %s %s'%(r[0],r[1],r[2],r[3],r[4])
    sub.append(s)

id_test = np.arange(len(sub))
submission = pd.DataFrame()
submission['id'] = id_test
submission['hotel_cluster'] = sub
submission.to_csv("submission_qianqian.csv", index=False) 
