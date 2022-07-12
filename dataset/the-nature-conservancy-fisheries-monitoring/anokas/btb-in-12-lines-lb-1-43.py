import pandas as pd
import glob
from scipy.misc import imread
from collections import defaultdict
g=pd.read_csv('../input/sample_submission_stg1.csv')['image']
s,t=zip(*[(imread(f).shape,f.split('/')[-2]) for f in glob.glob('../input/train/*/*.jpg')])
p=defaultdict(dict)
for s,m in pd.DataFrame({'s':s,'t':t}).groupby('s'): 
    for i,v in m.groupby('t').mean().to_dict()['s'].items():p[s][i]=v
s=(pd.DataFrame([p[imread('../input/test_stg1/'+f).shape] for f in g]).fillna(0))*0.99+0.005
s['image']=g
s.to_csv('btb.csv',index=False)