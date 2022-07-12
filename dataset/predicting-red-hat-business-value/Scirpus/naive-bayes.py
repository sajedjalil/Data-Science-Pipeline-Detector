import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName)['outcome'].mean().reset_index()
    grpCount = data1.groupby(columnName)['outcome'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.outcome
    grpOutcomes = grpOutcomes[grpOutcomes.cnt > 39]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
        x = x + np.random.normal(0, .05, x.shape[0])
    return x.fillna(x.mean())


def main():
    directory = '../input/'
    people = pd.read_csv(directory+'people.csv',
                         parse_dates=['date'])
    train = pd.read_csv(directory+'act_train.csv',
                        parse_dates=['date'])
    train = pd.merge(train, people,
                     how='left',
                     on='people_id',
                     suffixes=('_train', '_people'),
                     left_index=True)
    train['year'] = train['date_train'].dt.year
    train['month'] = train['date_train'].dt.month
    train.drop('date_train', axis=1, inplace=True)
    train['year_people'] = train['date_people'].dt.year
    train['month_people'] = train['date_people'].dt.month
    train.drop('date_people', axis=1, inplace=True)
    lootrain = pd.DataFrame()
    for col in train.columns:
        if(col != 'outcome' and col != 'people_id' and col != 'activity_id'):
            lootrain[col] = LeaveOneOut(train, train, col, True).values
    lootrain['outcome'] = train['outcome'].values
    features = lootrain.columns[1:-1]
    goodfeatures = []
    for col in features:
        score = roc_auc_score(lootrain.outcome,
                              lootrain[col])
        if(score >= .65):
            print(col, score)
            goodfeatures.append(col)
    print(goodfeatures)
    # uniquepeople = np.unique(train['people_id'])
    # for i in range(10):
    #     chc = np.random.choice(uniquepeople, int(len(uniquepeople)*.9))
    #     clf = GaussianNB()
    #     clf.fit(lootrain.loc[np.in1d(train.people_id.values, chc)][goodfeatures],
    #            lootrain.loc[np.in1d(train.people_id.values, chc)].outcome)
    #     probs = \
    #         clf.predict_proba(
    #             lootrain.loc[~np.in1d(train.people_id.values, chc)][goodfeatures])[:, 1]
    #     print(roc_auc_score(lootrain.loc[~np.in1d(train.people_id.values, chc)].outcome, probs))
    clf = GaussianNB()
    clf.fit(lootrain[goodfeatures], lootrain.outcome)
    probs = clf.predict_proba(lootrain[goodfeatures])[:, 1]
    print(roc_auc_score(lootrain.outcome, probs))
    test = pd.read_csv(directory+'act_test.csv',
                       parse_dates=['date'])
    test = pd.merge(test, people,
                    how='left',
                    on='people_id',
                    suffixes=('_train', '_people'),
                    left_index=True)
    test['year'] = test['date_train'].dt.year
    test['month'] = test['date_train'].dt.month
    test.drop('date_train', axis=1, inplace=True)
    test['year_people'] = test['date_people'].dt.year
    test['month_people'] = test['date_people'].dt.month
    test.drop('date_people', axis=1, inplace=True)
    activity_id = test.activity_id.values
    test.drop('activity_id', inplace=True, axis=1)
    test['outcome'] = 0
    lootest = pd.DataFrame()
    for col in goodfeatures:
        lootest[col] = LeaveOneOut(train, test, col, False).values
    probs = clf.predict_proba(lootest[goodfeatures])[:, 1]
    submission = pd.DataFrame({'activity_id': activity_id,
                               'outcome': probs})
    submission.to_csv('nbsubmission.csv', index=False)


if __name__ == "__main__":
    print('Started')
    main()
    print('Finished')
