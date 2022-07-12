import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from ml_metrics import mapk

dtypes = {'ad_id': np.float32, 'clicked': np.int8}

print('Load train clicks file...')
train = pd.read_csv("../input/clicks_train.csv", dtype=dtypes)
print(train.head()) # display_id ad_id clicked
print(train.shape) # (87141731, 3)

print('Ad position in display...')
train = train.iloc[:100000]
train['adposition'] = train.apply(lambda x: 
        train[train.display_id == x.values[0]].iloc[:,1].tolist().index(x.values[1]),
        axis=1)
#train.groupby('display_id').ad_id.apply(list)
print(train.head(20)) # display_id ad_id clicked
print(train.shape) # (87141731, 4)

print('Get advert position...')
ad_meanpos = train.groupby('ad_id').adposition.agg(['mean']).reset_index()
print(ad_meanpos.shape)
print(ad_meanpos.head(20))
mean_pos= train.adposition.mean()
print('Mean position: %0.4f' % mean_pos)
mean_clickedpos= train[train.clicked==1].adposition.mean()
print('Mean clicked position: %0.4f' % mean_clickedpos)

print('Append position series')
#reg = 8
#ad_meanpos['pos'] = (ad_prob['sum'] + reg*mean_clicked) / (reg + ad_prob['count'])
#ad_prob.drop(['count', 'sum', 'mean'], axis = 1, inplace = True)
print(ad_meanpos.shape)
print(ad_meanpos.head())

print('Train and cval sets...')
ids = train.display_id.unique()
cvids = np.random.choice(ids, size=len(ids)//10, replace=False)
valid = train[train.display_id.isin(cvids)]
train = train[~train.display_id.isin(cvids)]
print (train.shape, valid.shape) # (78428046, 3) (8713685, 3)

print('Get advert probability...')
ad_prob = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()
print(ad_prob.shape) # (467528, 4)
print(ad_prob.head())
mean_clicked = train.clicked.mean()
print('Mean probability: %0.5f' % mean_clicked) # 0.19364

totalclicks = ad_prob['sum'].sum()
sortprob = ad_prob.sort_values(['sum'], ascending=False)
print(sortprob.head())
for m in [100, 1000, 5000]:#, 10000, 20000, 50000, 100000]:
    mclicks = sortprob['sum'].iloc[:m].sum()
    print('%d most probable ads, %d clicks (%0.5f)' % (m, mclicks,
                1.*mclicks/totalclicks))

print('Append probability series')
reg = 12
ad_prob['prob'] = (ad_prob['sum'] + reg*mean_clicked) / (reg + ad_prob['count'])
ad_prob.drop(['count', 'sum', 'mean'], axis = 1, inplace = True)
print(ad_prob.shape)
print(ad_prob.head())

print('Merge valid set with adprob...')
valid = valid.merge(ad_prob, how='left')
valid.prob.fillna(mean_clicked, inplace=True)
valid.sort_values(['display_id','prob'], inplace=True, ascending=[True,False])
print(valid.head())
print(valid.shape)

print('Cross validation clicks...')
yval = valid[valid.clicked==1].ad_id
#print(yval.head(20))
yval = [[v] for v in yval.values] # list of lists of clicked ads for every display
print(yval[:5])
print('Cross validation probabilities...')
#print(valid.head(20))
pval = valid.groupby('display_id').ad_id.apply(list)
#print(pval.head(20))
pval = pval.values.tolist()
#print(pval[:5])
print ('Cval MAP@12: %0.5f ' % mapk(yval, pval, k=12))

print('Advert details...')
adetails = pd.read_csv('../input/promoted_content.csv')
print(adetails.shape) #
print(adetails.head())

print('Merge')
adetails = adetails.merge(ad_prob,  how='left')
print(adetails .shape) # 16874593
print(adetails .head())
print(adetail.info())

del train, valid, sortprob, events, train_disp
gc.collect()

print('Load test clicks...')
test = pd.read_cs("../input/clicks_test.csv")
print(test.shape)
print(test.head())
print('Merge with ad probability and fill NA...')
test = test.merge(ad_prob, how='left')
test.likelihood.fillna(mean_clicked, inplace=True)
print(test.head())

print('Sort ad probabilituess for every display...')
test.sort_values(['display_id','prob'], inplace=True, ascending=False)
print(test.head())
print('Group ad_id by display...')
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
print(subm.head())
print('Submit...')
subm.to_csv("subm.csv", index=False)

del test
gc.collect()