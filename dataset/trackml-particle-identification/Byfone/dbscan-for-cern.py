import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from trackml.dataset import load_event
from sklearn import cluster, preprocessing
import glob

#https://www.kaggle.com/the1owl/the-martian

train = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/train_1/**'))])
test = np.unique([p.split('-')[0] for p in sorted(glob.glob('../input/test/**'))])
det = pd.read_csv('../input/detectors.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print(len(train), len(test), len(det), len(sub))

scl = preprocessing.StandardScaler()
dbscan = cluster.DBSCAN(eps=0.00715, min_samples=1, algorithm='auto', n_jobs=-1)
df_test = []
for e in test:
    hits, cells = load_event(e, parts=['hits', 'cells'])
    hits['event_id'] = int(e[-9:])
    cells = cells.groupby(by=['hit_id'])['ch0', 'ch1', 'value'].agg(['mean']).reset_index()
    cells.columns = ['hit_id', 'ch0', 'ch1', 'value']
    hits = pd.merge(hits, cells, how='left', on='hit_id')
    col = [c for c in hits.columns if c not in ['event_id', 'hit_id', 'particle_id']]

    #https://www.kaggle.com/mikhailhushchyn/dbscan-benchmark
    x = hits.x.values
    y = hits.y.values
    z = hits.z.values
    r = np.sqrt(x**2 + y**2 + z**2)
    hits['x2'] = x/r
    hits['y2'] = y/r
    r = np.sqrt(x**2 + y**2)
    hits['z2'] = z/r
    hits['particle_id'] = dbscan.fit_predict(scl.fit_transform(hits[['x2', 'y2', 'z2']].values))
    
    df_test.append(hits[['event_id','hit_id','particle_id']].copy())
    print(e, len(hits['particle_id'].unique()))
    #break
    
df_test = pd.concat(df_test, ignore_index=True)

sub = pd.merge(sub, df_test, how='left', on=['event_id','hit_id'])
sub['track_id'] = sub['particle_id'] + 1
sub[['event_id','hit_id','track_id']].to_csv('submission-001.csv', index=False)