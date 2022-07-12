import pandas as d;d.np.random.seed(42);t,r=[d.read_csv('../input/t'+f+'.csv',index_col=0)for f in['rain','est']];*c,=t;M=c.pop(146);*_,T=c;*c,_=c
from sklearn import *;u=svm.NuSVC(.59,'poly',4,'auto',.053,1,1);q=discriminant_analysis.QuadraticDiscriminantAnalysis(0.111)
for m in range(512):i,j=t[M]==m,r[M]==m;x,z=t[i],r[j];f=[l for l in c if x[l].std()>1.1];y=x[T];u.fit(x[f],y);q.fit(x[f],y);r.loc[j,T]=.8*u.predict_proba(z[f])[:,1]+.2*q.predict_proba(z[f])[:,1]
r[[T]].to_csv('submission.csv',float_format='%.1f')