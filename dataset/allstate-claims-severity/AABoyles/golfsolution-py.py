import pandas as p
import statsmodels.formula.api as s
e=p.read_csv('../input/test.csv').dropna(1)
p.DataFrame({'id':e['id'].values,'loss':s.ols('loss~'+'+'.join([c for c in e.columns if 'cont' in c]),data=p.read_csv('../input/train.csv')).fit().predict(e)}).to_csv('o.csv',index=False)
