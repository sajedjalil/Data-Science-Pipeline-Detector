import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    print('Started!')
    print('Split Train into 90:10 split')
    print('& see how plain average does')
    train = pd.read_csv('../input/train.csv')
    test = train[::10].copy()
    testtargets = test.TARGET.values
    test.drop('TARGET', inplace=True, axis=1)
    train = train[~train.ID.isin(test.ID)].copy()
    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    trainaverager = pd.DataFrame({'ID': train.ID.values})
    testaverager = pd.DataFrame({'ID': test.ID.values})
    features = train.columns[1:-1]
    for i in features:
        trainaverager[i] = train[i].values
        testaverager[i] = test[i].values
        x = pd.DataFrame({i: train[i].values, 'TARGET': train.TARGET.values})
        x = x.groupby(i)['TARGET'].mean().reset_index()
        trainaverager = trainaverager.merge(x, how='left', on=i)
        trainaverager.drop(i, inplace=True, axis=1)
        trainaverager.rename(columns={'TARGET': i}, inplace=True)
        testaverager = testaverager.merge(x, how='left', on=i)
        testaverager.drop(i, inplace=True, axis=1)
        testaverager.rename(columns={'TARGET': i}, inplace=True)
    trainaverager['TARGET'] = train.TARGET

    trainaverager.fillna(trainaverager.mean(), inplace=True)
    testaverager.fillna(testaverager.mean(), inplace=True)

    features = trainaverager.columns[1:-1]
    secondtrainaverager = trainaverager.copy()
    secondtestaverager = testaverager.copy()
    for i in features:
        otherfeatures = list(set(features).difference(set([i])))
        secondtrainaverager[i] = trainaverager[otherfeatures].mean(axis=1)
        secondtestaverager[i] = testaverager[otherfeatures].mean(axis=1)
    secondtrainaverager['TARGET'] = train.TARGET

    remove = []
    c = secondtrainaverager.columns
    for i in range(len(c)-1):
        v = secondtrainaverager[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, secondtrainaverager[c[j]].values):
                remove.append(c[j])

    secondtrainaverager.drop(remove, axis=1, inplace=True)
    secondtestaverager.drop(remove, axis=1, inplace=True)

    remove = []
    for col in secondtrainaverager.columns:
        if secondtrainaverager[col].std() == 0:
            remove.append(col)
    secondtrainaverager.drop(remove, axis=1, inplace=True)
    secondtestaverager.drop(remove, axis=1, inplace=True)
    features = secondtrainaverager.columns[1:-1]
    print('Visible ROC:', roc_auc_score(train.TARGET,
          secondtrainaverager[features].mean(axis=1)))
    print('Blind ROC:', roc_auc_score(testtargets,
          secondtestaverager[features].mean(axis=1)))
    print('Finished')
