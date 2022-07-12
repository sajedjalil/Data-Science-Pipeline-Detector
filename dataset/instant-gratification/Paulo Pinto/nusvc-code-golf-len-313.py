import sklearn.svm as S, pandas as P;s=S.NuSVC(.6,'poly',4,.02,.05,1,1);t,r=[P.read_csv('../input/t'+f+'.csv',index_col=0)for f in['rain','est']];*c,=t;M=c.pop(146);*_,T=c;*c,_=c
for i in range(512):x,j=t[t[M]==i],r[M]==i;s.fit(x[c],x[T]);r.loc[j,T]=s.predict_proba(r[j][c])[:,1]
r[[T]].to_csv('submission.csv')