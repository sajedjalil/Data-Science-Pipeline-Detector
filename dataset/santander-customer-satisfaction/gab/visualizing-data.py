import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Script takes a while to run...")

train=pd.read_csv('../input/train.csv',index_col=0)

## Replacing missing values with mode
train.replace(-999999,2)
train.replace(9999999999,0)

y=train.TARGET
train=train.iloc[:,:-1]


#Removing constant features
keep=train.std(axis=0)>0
X=train.loc[:,keep]

#Removing perfectly correlated features (equal of scalar multiples)
from scipy.stats import pearsonr

duplicated=[]
ncols=X.shape[1]

for i in range(0,ncols-1):
    for j in range(i+1,ncols):
        if pearsonr(X.iloc[:,i],X.iloc[:,j])[0]>=1 : 
            duplicated.append(X.iloc[:,j].name)

X=X.drop(duplicated,axis=1)

#Log transforming features with range>=1000

ranges=X.max()-X.min()
logmap= lambda x : np.copysign(np.log10(abs(x)+1),x)
Xlog=X.loc[:,ranges>1000].applymap(logmap)
X.loc[:,Xlog.columns]=Xlog

#saldo_medio_var29_hace3 doesn't have range>1000 but let's transform it 
# for consistency with similar features

X.loc[:,'saldo_medio_var29_hace3']=X['saldo_medio_var29_hace3'].map(logmap)


#Plotting the data
#Change these values to the indices of the first and last features you want to plot
#kaggle doesn't allow plotting all features since image size is too large.
fromi=0
toi=200


f, axarr = plt.subplots((toi-fromi),2,figsize=(6,1.5*(toi-fromi)))
f.tight_layout()

X1=X.iloc[:,fromi:toi].columns

for i,col in enumerate(X1):
    toplot=[X[col][y==0].values,X[col][y==1].values]
    axarr[i,0].boxplot(toplot,vert=False,labels=['0','1'])
    axarr[i,1].hist(toplot,normed=True,label=['0','1'])
    axarr[i,0].set_title(col,loc='right')
    axarr[i,1].set_yscale('log')

axarr[0,1].legend();  
plt.subplots_adjust(wspace=0.3,hspace=0.5)

plt.savefig('features.png')