import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB

def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    
dtypes = {'ad_id': np.float32, 'clicked': np.int8}

print('Load train clicks...')
train = pd.read_csv("../input/clicks_train.csv", dtype=dtypes)
print(train.head()) # display_id ad_id clicked
print(train.shape) # (87141731, 3)

print('Get advert likelihood')
ad_likelihood = train.groupby('ad_id').clicked.agg(['count','sum','mean']).reset_index()
print(ad_likelihood.shape) # 478950 adverts
print(ad_likelihood.head())
M = train.clicked.mean()
print('Mean likelihood: %0.5f' % M)

totalclicks = ad_likelihood['sum'].sum()
slikelih = ad_likelihood.sort_values(['sum'], ascending=False)
print(slikelih.head())
for m in [100, 1000, 5000, 10000, 20000, 50000, 100000]:
    mclicks = slikelih['sum'].iloc[:m].sum()
    print('%d most probable ads, %d clicks (%0.5f)' % (m, mclicks,
                1.*mclicks/totalclicks))

print('Append likelihood series')
ad_likelihood['likelihood'] = (ad_likelihood['sum'] + 12*M) / (12 + ad_likelihood['count'])
print(ad_likelihood.shape)
print(ad_likelihood.head())

print('Events...')
events = pd.read_csv('../input/events.csv', usecols=['display_id', 'document_id',
                                                            'timestamp', 'platform'])
print('Events shape: ', events.shape) # (23120126, 6)
print('Columns', events.columns.tolist())
print(events.head())

print('Group by display')
train_disp = train.groupby('display_id')['ad_id'].sum().merge(events['display_id'], 
                                                                    how='left')
print(train_disp.shape) # 16874593
print(train_disp.head())

del train, slikelih, events, train_disp
gc.collect()

print('Load test clicks...')
test = pd.read_cs("../input/clicks_test.csv")
print(test.shape)
print(test.head())
print('Merge with ad likelihoods and fill NA...')
test = test.merge(ad_likelihood, how='left')
test.likelihood.fillna(M, inplace=True)
print(test.head())

print('Sort ad likelihoods for every display...')
test.sort_values(['display_id','likelihood'], inplace=True, ascending=False)
print(test.head())
print('Group ad_id by display...')
subm = test.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
print(subm.head())
print('Submit...')
subm.to_csv("subm.csv", index=False)

del test
gc.collect()