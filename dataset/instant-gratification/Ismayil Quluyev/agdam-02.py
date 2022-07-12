import pandas as d;d.np.random.seed(42);t,r=[d.read_csv('../input/t'+f+'.csv',index_col=0)for f in['rain','est']];*c,=t;M=c.pop(146);*_,T=c;*c,_=c
from sklearn import svm as S,neighbors as N;u=S.NuSVC(.6,'poly',4,'auto',.08,1,1);s=S.SVC(.6,'poly',4,'auto',.08,1,1);k=N.KNeighborsClassifier(15,p=2.9)
for i in range(512):x,j=t[t[M]==i],r[M]==i;u.fit(x[c],x[T]);s.fit(x[c],x[T]);k.fit(x[c],x[T]);p=r[j][c];r.loc[j,T]=.6*u.predict_proba(p)[:,1]+.3*s.predict_proba(p)[:,1]+.1*k.predict_proba(p)[:,1]
r[[T]].to_csv('submission.csv')