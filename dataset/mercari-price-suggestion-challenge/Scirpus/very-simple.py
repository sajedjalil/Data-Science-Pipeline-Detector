import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def SplitCategoryName(data):
    x = data['category_name'].str.split('/')
    x = np.array(x)
    maincat = []
    intcat = []
    childcat = []
    for a in range(x.shape[0]):
        if(isinstance(x[a], list)):
            maincat.append(x[a][0])
            intcat.append(x[a][1])
            childcat.append(x[a][2])
        else:
            maincat.append('')
            intcat.append('')
            childcat.append('')
    data['MainCat'] = maincat
    data['IntCat'] = intcat
    data['ChildCat'] = childcat
    data.drop('category_name',inplace=True,axis=1)
    return data

def LeaveOneOut(data1, data2, groupcolumns, columnName, useLOO=False, addNoise=False, cut=1, noise=0.0005):
    features = list([])
    for a in groupcolumns:
        features.append(a)
    features.append(columnName)
    grpOutcomes = data1.groupby(features)['price'].mean().reset_index()
    grpCount = data1.groupby(features)['price'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.price
    outcomes = data2['price'].values
    if(useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > cut]
    grpOutcomes.drop('cnt', inplace=True, axis=1)

    x = pd.merge(data2[features+list(['price'])], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=features,
                 left_index=True)['price']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
        if(addNoise):
            x = x + np.random.normal(0, noise, x.shape[0])
    return x.fillna(x.mean()).values


strdirectory = '../input/'

cats = list(['loo_MainCat',
        'loo_IntCat',
        'loo_ChildCat',
        'loo_item_condition_id',
        'loo_brand_name',
        'loo_shipping'])


pricecats = list(['loo_MainCat',
                 'loo_IntCat',
                 'loo_ChildCat',
                 'loo_item_condition_id',
                 'loo_brand_name',
                 'loo_shipping',
                 'price'])

highcardinality = list(['MainCat',
                        'IntCat',
                        'ChildCat',
                        'item_condition_id',
                        'brand_name',
                        'shipping'])

train = pd.read_table(strdirectory+'train.tsv')
train['price'] = np.log1p(train.price.values)
test = pd.read_table(strdirectory+'test.tsv')
test['price'] = np.nan
train = SplitCategoryName(train)
test = SplitCategoryName(test)
for c in highcardinality:
    test.insert(1,'loo_'+c,
                  LeaveOneOut(train[(train.price>5.5)&(train.price<6)],
                              test,
                              list([]),
                              c,
                              useLOO=False,
                              addNoise=False,
                              cut=100, noise=0.0))
    train.insert(1,'loo_'+c,
                  LeaveOneOut(train[(train.price>5.5)&(train.price<6)],
                              train,
                              list([]),
                              c,
                              useLOO=True,
                              addNoise=False,
                              cut=100, noise=0.0))


train.drop(highcardinality,inplace=True,axis=1)
test.drop(highcardinality,inplace=True,axis=1)
others = ['name','item_description']
train.drop(others,inplace=True,axis=1)
test.drop(others,inplace=True,axis=1)

test['price'] = np.expm1(test[test.columns[1:-1]].mean(axis=1))
test[['test_id', 'price']].to_csv('simples.csv', index=False)