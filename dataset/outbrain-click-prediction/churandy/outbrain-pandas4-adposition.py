import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import xgboost as xgb
from ml_metrics import mapk

np.random.seed(27)

def fitClassifier(clf, train, valid):
    Xtrain = train.adposition.values.reshape((-1, 1))
    ytrain = train.clicked.values
    #print(X.shape, y.shape)
    Xval = valid.adposition.values.reshape((-1, 1))
    yval = valid.clicked.values
    #X = np.reshape(X, (-1, 1))
    clf.fit(Xtrain, ytrain)
    print('Train and cval scores: %0.4f, %0.4f' % (clf.score(Xtrain, ytrain),
                                                clf.score(Xval, yval)))
    #print(clf.predict_proba(Xval)[:10])
    # Pick the best threshold out-of-fold
    print('Pick best threshold...')
    predprobs = clf.predict_proba(Xval)[:,1]
    thresholds = np.linspace(0.01, 0.99, 50)
    #mcc = np.array([matthews_corrcoef(yval, predprobs>thr) for thr in thresholds])
    acc = np.array([accuracy_score(yval, predprobs>thr) for thr in thresholds])
    #print(acc)
    best_threshold = thresholds[np.argmax(acc)]
    auc = roc_auc_score(yval, predprobs)
    print('Max accur: %0.5f, AUC: %0.5f (threshold %0.2f)' % 
                                    (np.max(acc), auc, best_threshold))
    #print('Cross validation clicks...')
    yval = valid[valid.clicked==1].ad_id
    #print(yval.head(20))
    yval = [[v] for v in yval.values] # list of lists of clicked ads for every display
    #print(yval[:5])
    #print('Cross validation probabilities...')
    #print(valid.head(20))
    pval = valid.groupby('display_id').ad_id.apply(list)
    #print(pval.head(20))
    pval = pval.values.tolist()
    #print(pval[:5])
    print ('Cval MAP@12: %0.5f ' % mapk(yval, pval, k=12))
    return predprobs

dtypes = {'ad_id': np.float32, 'clicked': np.int8}

print('Load train clicks file...')
train = pd.read_csv("../input/clicks_train.csv", dtype=dtypes)
print(train.head()) # display_id ad_id clicked
print(train.shape) # (87141731, 3)

print('Ad position in display...')
train = train.iloc[:10000]
train['adposition'] = train.apply(lambda x: 
        train[train.display_id == x.values[0]].iloc[:,1].tolist().index(x.values[1]),
        # / train[train.display_id == x.values[0]].iloc[:,1].tolist()
        axis=1)
#train.groupby('display_id').ad_id.apply(list)
print(train.head(20)) # display_id ad_id clicked
print(train.shape) # (87141731, 4)
mean_pos= train.adposition.mean()
print('Mean position: %0.4f' % mean_pos)
mean_clickedpos= train[train.clicked==1].adposition.mean()
print('Mean clicked position: %0.4f' % mean_clickedpos)
#print(train.info())
#print(train.describe())

print('Get advert position...')
ad_meanpos = train.groupby('ad_id').adposition.agg(['mean']).reset_index()
print(ad_meanpos.shape)
print(ad_meanpos.head(20))

print('Append position series')
#reg = 8
#ad_meanpos['pos'] = (ad_prob['sum'] + reg*mean_clicked) / (reg + ad_prob['count'])
#ad_prob.drop(['count', 'sum', 'mean'], axis = 1, inplace = True)
#print(ad_meanpos.shape)
#print(ad_meanpos.head())

print('Train and cval sets...')
ids = train.display_id.unique()
cvids = np.random.choice(ids, size=len(ids)//10, replace=False)
valid = train[train.display_id.isin(cvids)]
train = train[~train.display_id.isin(cvids)]
print (train.shape, valid.shape) # (78428046, 3) (8713685, 3)

print('Train with ad positions...')
fitClassifier(GaussianNB(), train, valid)
fitClassifier(SGDClassifier(loss = 'log'), train, valid)
fitClassifier(RandomForestClassifier(n_estimators = 200, random_state=11), 
                                                        train, valid)
fp=fitClassifier(xgb.XGBClassifier(max_depth=5, base_score=0.2), train, valid)

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
                
print('Append fitted probs...')
valid['fittedprob'] = fp
print(valid.head())
print(valid.shape)

print('Append probability series')
reg = 12
ad_prob['prob'] = (ad_prob['sum'] + reg*mean_clicked) / (reg + ad_prob['count'])
ad_prob.drop(['count', 'sum', 'mean'], axis = 1, inplace = True)
print(ad_prob.shape)
print(ad_prob.head())

print('Merge valid set with adprob...')
valid = valid.merge(ad_prob, how='left')
valid.prob.fillna(mean_clicked, inplace=True)
print(valid.head())
print(valid.shape)

wfp = 0.03
valid['prob']= valid.prob + wfp*valid.fittedprob
print(valid.head())
print(valid.shape)

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
# 10k, 0: 0.59263, 0.001: 0.59245, 0.01: 0.59233, 0.1: 0.58949, 0.03: 0.57659
# 0.2: 0.58414

