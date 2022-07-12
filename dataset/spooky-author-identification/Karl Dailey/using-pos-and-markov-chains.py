############################################################
## imports
############################################################
from catboost import CatBoostClassifier
from collections import defaultdict
from datetime import datetime
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

############################################################
## other globals
############################################################
PATH = '../input/'
cols_to_drop = ['id', 'text', 'author']

############################################################
## function definitions
############################################################
## USE word_tokenize(x.decode('utf-8') FOR PYTHON 2.7!!!!!!!
tok_and_tag = lambda x: pos_tag(word_tokenize(x))

def update_tag_dictionary(tags, d, d2):
    for i,t in enumerate([('start','start')] + tags[:-1]):
        d[(t[1], tags[i][1])]=d[(t[1], tags[i][1])]+1
        d2[t[1]]=d2[t[1]]+1

## Use log prob since multiplying lots of small numbers will always be 0
def get_markov_prob(tags, d, ALL):
    log_prob = 0
    for i,t in enumerate([('start','start')] + tags[:-1]):
        if d[(t[1], tags[i][1])]:
            log_prob= log_prob + np.log(d[(t[1], tags[i][1])])
        else:
            log_prob= log_prob +np.log(ALL[(t[1], tags[i][1])])
    return log_prob

############################################################
## loading data
############################################################
print('Loading data...')
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
sample = pd.read_csv(PATH + 'sample_submission.csv')

############################################################
## Feature engineering
############################################################
print('Feature engineering...')
print('Problem: Want to create features that models an authors writing style by using the data')
print('Solution: Use markov chains on part of speach')


print(' markov chain...')
print('  pos tagging...')
train2 = train.copy()
train2['tags'] = train2['text'].apply(tok_and_tag)

test2 = test.copy()
test2['tags'] = test2['text'].apply(tok_and_tag)

ALL1, ALL2 = defaultdict(lambda: 0), defaultdict(lambda: 0.0)
train2['tags'].apply((lambda x: update_tag_dictionary(x, ALL1,ALL2)))
test2['tags'].apply((lambda x: update_tag_dictionary(x, ALL1,ALL2)))
ALL = defaultdict(int,{k:v/ALL2[k[0]] for k, v in ALL1.items()})

print('  calculating log probs for train...')
## I use these folds to limit the impact of data leakage.  technically speaking,
## I use the y's in training which is sort of cheating.

kf = model_selection.KFold(n_splits=3, shuffle=False, random_state=2017)
for dev_index, val_index in kf.split(train):
    EAP1,EAP2  = defaultdict(lambda: 0), defaultdict(lambda: 0.0)
    HPL1, HPL2 = defaultdict(lambda: 0), defaultdict(lambda: 0.0)
    MWS1, MWS2 = defaultdict(lambda: 0), defaultdict(lambda: 0.0)
    
    train2.loc[dev_index][train2.author=='EAP']['tags'].apply((lambda x: update_tag_dictionary(x, EAP1,EAP2)))
    EAP = defaultdict(int,{k:v/EAP2[k[0]] for k, v in EAP1.items()})
    train2.loc[dev_index][train2.author=='HPL']['tags'].apply((lambda x: update_tag_dictionary(x, HPL1,HPL2)))
    HPL = defaultdict(int,{k:v/HPL2[k[0]] for k, v in HPL1.items()})
    train2.loc[dev_index][train2.author=='MWS']['tags'].apply((lambda x: update_tag_dictionary(x, MWS1,MWS2)))
    MWS = defaultdict(int,{k:v/MWS2[k[0]] for k, v in MWS1.items()})

    train2.ix[val_index,'markov_prob_pos_EAP'] = train2.loc[val_index]['tags'].apply(( lambda x: get_markov_prob(x, EAP, ALL) - get_markov_prob(x, ALL, ALL) ))
    train2.ix[val_index,'markov_prob_pos_HPL'] = train2.loc[val_index]['tags'].apply(( lambda x: get_markov_prob(x, HPL, ALL) - get_markov_prob(x, ALL, ALL) ))
    train2.ix[val_index,'markov_prob_pos_MWS'] = train2.loc[val_index]['tags'].apply(( lambda x: get_markov_prob(x, MWS, ALL) - get_markov_prob(x, ALL, ALL) ))

print('  calculating log probs for test...')
EAP1,EAP2  = defaultdict(lambda: 0), defaultdict(lambda: 0.0)
HPL1, HPL2 = defaultdict(lambda: 0), defaultdict(lambda: 0.0)
MWS1, MWS2 = defaultdict(lambda: 0), defaultdict(lambda: 0.0)

train2[train2.author=='EAP']['tags'].apply((lambda x: update_tag_dictionary(x, EAP1,EAP2)))
EAP = defaultdict(int,{k:v/EAP2[k[0]] for k, v in EAP1.items()})
train2[train2.author=='HPL']['tags'].apply((lambda x: update_tag_dictionary(x, HPL1,HPL2)))
HPL = defaultdict(int,{k:v/HPL2[k[0]] for k, v in HPL1.items()})
train2[train2.author=='MWS']['tags'].apply((lambda x: update_tag_dictionary(x, MWS1,MWS2)))
MWS = defaultdict(int,{k:v/MWS2[k[0]] for k, v in MWS1.items()})

test2['markov_prob_pos_EAP'] = test2['tags'].apply(( lambda x: get_markov_prob(x, EAP, ALL) - get_markov_prob(x, ALL, ALL) ))
test2['markov_prob_pos_HPL'] = test2['tags'].apply(( lambda x: get_markov_prob(x, HPL, ALL) - get_markov_prob(x, ALL, ALL) ))
test2['markov_prob_pos_MWS'] = test2['tags'].apply(( lambda x: get_markov_prob(x, MWS, ALL) - get_markov_prob(x, ALL, ALL) ))

train['markov_prob_pos_EAP'] = train2['markov_prob_pos_EAP']
train['markov_prob_pos_HPL'] = train2['markov_prob_pos_HPL']
train['markov_prob_pos_MWS'] = train2['markov_prob_pos_MWS']

test['markov_prob_pos_EAP'] = test2['markov_prob_pos_EAP']
test['markov_prob_pos_HPL'] = test2['markov_prob_pos_HPL']
test['markov_prob_pos_MWS'] = test2['markov_prob_pos_MWS']


############################################################
## create X, Y
############################################################
label_encoder = LabelEncoder().fit(train.author)
y_train = label_encoder.transform(train.author.values)
x_train = train.drop(cols_to_drop+['author'], axis=1).values
x_test = test[ [c for c in train.columns if c not in cols_to_drop] ].values

print(x_train.shape,x_test.shape)


############################################################
## Model Parameters
############################################################
params_cat = {}
params_cat['random_seed']=1
params_cat['learning_rate']=0.1
params_cat['depth']=6
params_cat['l2_leaf_reg']=3
params_cat['loss_function']='MultiClass'
params_cat['classes_count']=3
params_cat['use_best_model']=False
params_cat['eval_metric'] = 'MultiClass'
params_cat['iterations']=500 

############################################################
## Run Models
############################################################
print('Run Models...')
print(' Ensemble Catboost Model...')
print('  catboost 1')

## By running 2 models with different random seeds, I shake out some of the variance
model_cat = CatBoostClassifier(**params_cat).fit(x_train, y_train)
y_hat_cat1 = model_cat.predict_proba(x_test)
print('  catboost 2')
params_cat['random_seed']=19
model_cat = CatBoostClassifier(**params_cat).fit(x_train, y_train)
y_hat_cat2 = model_cat.predict_proba(x_test)
y_hat_cat = y_hat_cat1 * .5 + y_hat_cat2 * .5

## a large number indicates a not so great model, I use this as a heuristic before pushing
## to the LB
print(sum(np.max(y_hat_cat, axis=1)<.6))

############################################################
## Results
############################################################
results = pd.DataFrame()
results = sample.copy()
y_hat_cat = pd.DataFrame(y_hat_cat, columns = ['EAP','HPL','MWS'])
results['EAP']=y_hat_cat['EAP']
results['HPL']=y_hat_cat['HPL']
results['MWS']=y_hat_cat['MWS']

## Save to CSV with timestamp
results.to_csv('submission.{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index = False)



