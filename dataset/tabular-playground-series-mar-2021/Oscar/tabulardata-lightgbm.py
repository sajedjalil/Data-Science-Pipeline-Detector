import pandas as pd
import numpy as np
import random
from lightgbm import LGBMClassifier

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    
def load_data(df,cv=False,target=False,**kwargs):
    num_samples = len(df)
    sample_size = len(args['cat_feats']) + len(args['num_feats'])
    dataset = np.zeros((num_samples,sample_size))
    idx = 0
    for c in args['cat_feats']:
        dataset[:,idx] = np.array([args['categories'][c][v] if v in args['categories'][c] else len(args['categories'][c]) for v in df[c]])
        idx += 1
    for n in args['num_feats']:
        dataset[:,idx] = np.array(df[n])
        idx += 1
    if target:
        targets = np.array(df['target'])
    else:
        targets = None
    
    if cv == False:
        return dataset, targets

    idx = [i for i in range(num_samples)]
    random.shuffle(idx)
    trainset = dataset[idx[0:int(num_samples*(1-kwargs['cv_percentage']))]]
    traintargets = targets[idx[0:int(num_samples*(1-kwargs['cv_percentage']))]]
    validset = dataset[idx[int(num_samples*(1-kwargs['cv_percentage'])):]]
    validtargets = targets[idx[int(num_samples*(1-kwargs['cv_percentage'])):]]
    return trainset, validset, traintargets, validtargets  

#Compute the class-based ROC metric
def compute_roc_auc(scores,labels):
    if scores.ndim == 2:
        scores = scores[:,1]
    pos_scores = scores[np.where(labels==1)]
    neg_scores = scores[np.where(labels==0)]
    thresholds = np.arange(np.max(scores),np.min(scores)-0.01,-0.001)
    fpr = []
    tpr = []
    for th in thresholds:
        fpr.append(len(np.where(neg_scores>=th)[0])/len(neg_scores))
        tpr.append(len(np.where(pos_scores>=th)[0])/len(pos_scores))
    roc = pd.DataFrame({'thresholds':thresholds,'tp':tpr,'fp':fpr})
    auc = np.trapz(roc['tp'],x=roc['fp'])
    return auc
    
args = {
    'cv_percentage': 0.1,
    'seed': 0,
    'num_leaves': 90,
    'max_depth': 30,
    'learning_rate': 0.1,
    'n_estimators': 200,
    }

random_init(**args)

train_data = pd.read_csv('../input/tabular-playground-series-mar-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-mar-2021/test.csv')
args['cat_feats'] = [c for c in np.sort(train_data.columns) if 'cat' in c]
args['num_feats'] = [c for c in np.sort(train_data.columns) if 'cont' in c]
args['categories'] = {c: {a:i for i,a in enumerate(np.unique(train_data[c]))} for c in args['cat_feats']}
trainset, validset, traintargets, validtargets = load_data(train_data,cv=True,target=True,**args)
args['num_mean'] = np.mean(trainset[:,len(args['cat_feats']):],axis=0)
args['num_std'] = np.std(trainset[:,len(args['cat_feats']):],axis=0)
testset, _ = load_data(test_data,cv=False,target=False,**args)

trainset[:,len(args['cat_feats']):] -= args['num_mean']
trainset[:,len(args['cat_feats']):] /= args['num_std']
validset[:,len(args['cat_feats']):] -= args['num_mean']
validset[:,len(args['cat_feats']):] /= args['num_std']
testset[:,len(args['cat_feats']):] -= args['num_mean']
testset[:,len(args['cat_feats']):] /= args['num_std']

lgb = LGBMClassifier(num_leaves=args['num_leaves'],max_depth=args['max_depth'],learning_rate=args['learning_rate'],n_estimators=args['n_estimators'],objective="binary")
lgb.fit(trainset,traintargets)
val_pred_lgb = lgb.predict_proba(validset)
test_pred_lgb = lgb.predict_proba(testset)
auc = compute_roc_auc(val_pred_lgb,validtargets)
print('Validation AUC: {0:.3f}'.format(auc))

out_df = pd.DataFrame(data={'id':test_data['id'],'target':test_pred_lgb[:,1]}).set_index('id',drop=True)
out_df.to_csv('submission.csv')
