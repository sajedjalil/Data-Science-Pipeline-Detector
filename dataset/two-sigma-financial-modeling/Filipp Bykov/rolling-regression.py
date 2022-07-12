import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm

# The "environment" is our interface for code competitions
env = kagglegym.make()
# We get our initial observation by calling "reset"
observation = env.reset()
# Get the train dataframe
train = observation.train

cols_to_use = ['technical_20','technical_30','technical_13','y']
excl = ['id', 'y', 'timestamp']
allcol = [c for c in train.columns if ((c in excl)|(c in cols_to_use))]
allcol1 = [c for c in allcol if not (c == 'y')]
train=train[allcol]

low_y_cut = -0.086093
high_y_cut = 0.093497

mean_values = train.median(axis=0)
#train.fillna(mean_values, inplace=True)

pred = np.array(train[cols_to_use])
tis=np.array(train.loc[:, 'timestamp'],dtype=int)
ids=np.array(train.loc[:, 'id'],dtype=int)
del train

predtab=np.zeros((max(tis)+1,max(ids)+1,pred.shape[1]))
predtab[:,:,:]=np.nan
for c in range(0,max(ids)+1) :
  sel = np.array(ids==c)
  predtab[tis[sel],c,:]=pred[sel,:]
del pred,tis,ids

gconst = [1,-1]
for iter in range(0,2):
    dt=gconst[0]*predtab[:-1,:,0:3]+gconst[1]*predtab[1:,:,0:3]
    trg=predtab[:-1,:,-1]
    ok=np.array((np.sum(np.isnan(dt),axis=2)==0)&np.isfinite(trg)&(trg<high_y_cut)&(trg>low_y_cut))
    met1=lm.LinearRegression()
    dt = dt[np.repeat(ok.reshape((ok.shape[0],ok.shape[1],1)),dt.shape[2],axis=2)].reshape(np.sum(ok),dt.shape[2])
    met1.fit (dt,trg[ok])
    r2 = met1.score(dt,trg[ok])
    dconst = met1.coef_
    print('Dconst=',dconst,' R=',np.sqrt(r2))
    
    dt=np.dot(predtab[:,:,0:3],dconst).reshape((predtab.shape[0],predtab.shape[1],1))
    dt=np.concatenate((dt[:-1,:,:],dt[1:,:,:]),axis=2)
    ok=np.array((np.sum(np.isnan(dt),axis=2)==0)&np.isfinite(trg)&(trg<high_y_cut)&(trg>low_y_cut))
    met1=lm.LinearRegression()
    dt = dt[np.repeat(ok.reshape((ok.shape[0],ok.shape[1],1)),dt.shape[2],axis=2)].reshape(np.sum(ok),dt.shape[2])
    met1.fit (dt,trg[ok])
    r2 = met1.score(dt,trg[ok])
    gconst = met1.coef_
    print('Gconst=',gconst,' R=',np.sqrt(r2))
del dt, trg, ok

def expandmas2 (mas,n):
    if (mas.shape[1]<=n):
        mas1=np.zeros((mas.shape[0],int(n*1.2+1)))
        for i in range(0,mas.shape[0]):
            mas1[i,:]=mas[i,-1]
        mas1[:,:mas.shape[1]]=mas
        mas = mas1
    return mas
def expandmas (mas,n,m):
    if (mas.shape[0]<=n):
        mas1=np.zeros((int(n*1.2+1),mas.shape[1],mas.shape[2]))
        mas1[:,:,:]=np.nan
        mas1[:mas.shape[0],:mas.shape[1],:]=mas
        mas = mas1
    if (mas.shape[1]<=m):
        mas1=np.zeros((mas.shape[0],int(m*1.2+1),mas.shape[2]))
        mas1[:,:,:]=np.nan
        mas1[:mas.shape[0],:mas.shape[1],:]=mas
        mas = mas1
    return mas

realhist = predtab.shape[0]
coef = np.zeros((1,realhist))
def trainrolling (tset):
    for t in tset :            
            s0=max(t-1,1)
            y=predtab[s0,:,-1]
            x=predtab[s0-1,:,-1]
            ok=np.array(np.isfinite(x)&np.isfinite(y)&(x>low_y_cut)&(x<high_y_cut)&(y<high_y_cut)&(y>low_y_cut))
#            ok=np.array(np.isfinite(x)&np.isfinite(y))
            if np.sum(ok)==0:
                    coef[0,t]=coef[0,t-1]
            else:                    
                    x1=x[ok]
                    y1=y[ok]
#                    alp1=0.65*(np.std(x1)+np.std(y1))*max(200,x1.shape[0])
                    alp1=np.std(np.concatenate((x1,y1)))*max(200,x1.shape[0])
                    x1=np.concatenate((x1,[alp1]))
                    y1=np.concatenate((y1,[alp1*coef[0,t-1]]))
                    coef[0,t]=np.sum(x1*y1)/np.sum(x1*x1)
            if t>=1:
                res = predtab[t-1,:,-1]*coef[0,t]
    return res,coef

reward=0
n = 0
rewards = []
t0 = 0
while True:
    test = observation.features[allcol1].copy()
#    test['id'] = observation.target.id 
#    test.fillna(mean_values, inplace=True)
    pred=np.array(test[cols_to_use[:-1]])
    maxts = int(max(test['timestamp']))    
    maxid = int(max(test['id']))
    predtab=expandmas (predtab,maxts,maxid)
    coef =expandmas2 (coef,maxts)

    resout = np.zeros((pred.shape[0]))
    for t in range(int(min(test['timestamp'])),maxts+1) :
        sel=np.array(test['timestamp']==t)
        ids=np.array(test.loc[sel,'id'],dtype=int)
                
        predtab[t,ids,0:pred.shape[1]]=pred[sel,:]
        if (t<1):
            continue
        old = predtab[t-1,ids,-1]
#        new = np.dot(predtab[t,ids,0:3]-predtab[t-1,ids,0:3],dconst)
        new = np.dot(predtab[t-1:t+1,ids,0:3],dconst)
        new = np.dot(new.T,gconst)
        old[np.isnan(old)]=new[np.isnan(old)]
        predtab[t-1,ids,-1]=old
        t0=int(min(t0,t-1))
        
        res,coef = trainrolling (range(t0+1,t+1))
        res = res[ids]
        res [np.isnan(res)]=0.
        resout[sel]=res
        t0=t            

    observation.target.y = resout.clip(low_y_cut, high_y_cut)
    observation.target.y = observation.target.y
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(np.mean(rewards))

    observation, reward, done, info = env.step(target)
#    print(reward)
    if done:
        break
    rewards.append(reward)
    n = n + 1
print(info)
