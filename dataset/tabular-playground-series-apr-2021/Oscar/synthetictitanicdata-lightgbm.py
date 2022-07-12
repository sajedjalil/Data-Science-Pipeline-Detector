import sys
import pandas as pd
import numpy as np
import random
from lightgbm import LGBMClassifier

#Initialise the random seeds
def random_init(**kwargs):
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    
def normalise_data(df):
    df.loc[df['Pclass'].isna(),'Pclass'] = 0
    df.loc[df['Sex'].isna(),'Sex'] = '*'
    df.loc[df['Age'].isna(),'Age'] = -1
    df.loc[df['SibSp'].isna(),'SibSp'] = -1
    df.loc[df['Parch'].isna(),'Parch'] = -1
    df.loc[df['Embarked'].isna(),'Embarked'] = '*'
    df.loc[df['Fare'].isna(),'Fare'] = -1
    df.loc[df['Ticket'].isna(),'Ticket'] = -1
    df['Ticket'] = [''.join(c for c in t if c.isdigit()) if str(t)==t else str(t) for t in df['Ticket']]
    df['Ticket'] = [int(t) if len(t)>0 else -1 for t in df['Ticket']]
    return df
    
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
        targets = np.array(df['Survived'])
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

def compute_results(predictions,labels):
    if predictions.ndim == 2:
        predictions = predictions[:,1]
    thresholds = np.arange(1.0,-0.00001,-0.001)
    fpr = []
    tpr = []
    acc = []
    for th in thresholds:
        tp = np.sum((predictions >= th) * labels)
        tn = np.sum((predictions < th) * (1-labels))
        fp = np.sum((predictions >= th) * (1-labels))
        fn = np.sum((predictions < th) * labels)
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(tn+fp))
        acc.append((tp+tn)/(tp+tn+fp+fn))
    results = pd.DataFrame({'thresholds':thresholds,'tpr':tpr,'fpr':fpr,'acc':acc})
    return results
    
args = {
    'cv_percentage': 0.1,
    'seed': 0,
    'num_leaves': 90,
    'max_depth': 30,
    'learning_rate': 0.1,
    'n_estimators': 200,
    }

random_init(**args)

train_data = normalise_data(pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv'))
test_data = normalise_data(pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv'))
args['cat_feats'] = ['Pclass','Sex','Embarked']
args['num_feats'] = ['Age','SibSp','Parch','Ticket','Fare']
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
results = compute_results(val_pred_lgb,validtargets)
th = results.loc[results['acc']==np.max(results['acc'])]['thresholds'].values[0]
acc = 100*results.loc[results['acc']==np.max(results['acc'])]['acc'].values[0]
auc = np.trapz(results['tpr'],x=results['fpr'])
pos = 100*sum(test_pred_lgb[:,1] >= th)/test_pred_lgb.shape[0]
print('Validation AUC: {0:.3f}, validation accuracy: {1:.2f}%@{2:.3f}, test survival rate: {3:.2f}%'.format(auc,acc,th,pos))

out_df = pd.DataFrame(data={'PassengerId':test_data['PassengerId'],'Survived':(test_pred_lgb[:,1] >= th).astype(int)}).set_index('PassengerId',drop=True)
out_df.to_csv('submission.csv')
